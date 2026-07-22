# -*- coding: utf-8 -*-
"""Test MCSF-ResUNet on ACDC volumes without saving predictions by default."""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, zoom

from networks.mcsf_resunet import MCSFResUNet

try:
    from medpy.metric.binary import dc as medpy_dc
    from medpy.metric.binary import hd95 as medpy_hd95
except Exception:
    medpy_dc = None
    medpy_hd95 = None


CLASS_NAMES = ["Background", "Right ventricle", "Myocardium", "Left ventricle"]
IMAGENET_MEAN = np.asarray(
    [0.485, 0.456, 0.406], dtype=np.float32
).reshape(3, 1, 1)
IMAGENET_STD = np.asarray(
    [0.229, 0.224, 0.225], dtype=np.float32
).reshape(3, 1, 1)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Invalid boolean value: {}".format(value))


def setup_logger(output_dir):
    logger = logging.getLogger("mcsf_acdc_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(output_dir / "test.log", mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def split_directory(data_root, split):
    root = Path(data_root)
    aliases = {
        "test": ["test", "testing"],
        "valid": ["valid", "val", "validation"],
        "val": ["val", "valid", "validation"],
    }
    for name in aliases.get(split, [split]):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / split


def list_file_path(list_dir, split):
    directory = Path(list_dir)
    candidates = [directory / "{}.txt".format(split)]
    if split in {"valid", "val"}:
        candidates.extend([directory / "valid.txt", directory / "val.txt"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_split_names(data_root, list_dir, split):
    list_path = list_file_path(list_dir, split)
    if list_path is not None:
        with open(str(list_path), "r") as handle:
            return [line.strip() for line in handle if line.strip()]
    directory = split_directory(data_root, split)
    files = []
    for pattern in ["*.h5", "*.npz", "*.npy"]:
        files.extend(sorted(directory.glob(pattern)))
    return [path.name for path in files]


def find_sample_path(data_root, split, name):
    root = Path(data_root)
    directory = split_directory(data_root, split)
    raw = Path(name)
    bases = [raw] if raw.is_absolute() else [directory / raw, root / raw]
    for base in bases:
        if base.is_file():
            return base
        for suffix in ["", ".h5", ".npz", ".npy", ".npy.h5"]:
            candidate = Path(str(base) + suffix)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "Cannot find sample '{}' for split '{}'".format(name, split)
    )


def choose_key(keys, preferred):
    lower = {key.lower(): key for key in keys}
    for name in preferred:
        if name.lower() in lower:
            return lower[name.lower()]
    for key in keys:
        key_lower = key.lower()
        if any(name.lower() in key_lower for name in preferred):
            return key
    return None


def load_image_label(path):
    suffix = path.suffix.lower()
    if suffix == ".h5":
        with h5py.File(str(path), "r") as handle:
            keys = list(handle.keys())
            image_key = choose_key(keys, ["image", "img", "data"])
            label_key = choose_key(keys, ["label", "mask", "seg"])
            if image_key is None or label_key is None:
                raise KeyError(
                    "Cannot locate image/label in {}. keys={}".format(path, keys)
                )
            image = handle[image_key][()]
            label = handle[label_key][()]
    elif suffix == ".npz":
        data = np.load(str(path), allow_pickle=True)
        keys = list(data.keys())
        image_key = choose_key(keys, ["image", "img", "data"])
        label_key = choose_key(keys, ["label", "mask", "seg"])
        if image_key is None or label_key is None:
            if len(keys) < 2:
                raise KeyError(
                    "Cannot locate image/label in {}. keys={}".format(path, keys)
                )
            image_key, label_key = keys[0], keys[1]
        image = data[image_key]
        label = data[label_key]
    elif suffix == ".npy":
        array = np.load(str(path), allow_pickle=True)
        if not (isinstance(array, np.ndarray) and array.dtype == object):
            raise ValueError("Plain .npy is unsupported: {}".format(path))
        obj = array.item()
        image = obj.get("image", obj.get("img", obj.get("data")))
        label = obj.get("label", obj.get("mask", obj.get("seg")))
        if image is None or label is None:
            raise KeyError("Cannot locate image/label in {}".format(path))
    else:
        raise ValueError("Unsupported file type: {}".format(path))

    image = np.asarray(image)
    label = np.asarray(label)
    if image.ndim == 2:
        image = image[None, ...]
        label = label[None, ...]
    if image.ndim != 3 or image.shape != label.shape:
        raise ValueError(
            "Expected matching 3D image/label, got {} and {} in {}".format(
                image.shape, label.shape, path
            )
        )
    return image.astype(np.float32), label.astype(np.uint8)


def resize_numpy(array, output_size, order):
    if tuple(array.shape[-2:]) == tuple(output_size):
        return array
    factors = (
        float(output_size[0]) / float(array.shape[-2]),
        float(output_size[1]) / float(array.shape[-1]),
    )
    return zoom(array, factors, order=order)


def robust_mri_to_unit(image, z_clip=3.0):
    image = image.astype(np.float32)
    mean = float(image.mean())
    std = float(image.std())
    if std > 1e-6:
        image = (image - mean) / std
    else:
        image = image - mean
    z_clip = max(float(z_clip), 1e-3)
    image = np.clip(image, -z_clip, z_clip)
    return ((image + z_clip) / (2.0 * z_clip)).astype(np.float32)


def prepare_slice(image_slice, img_size, z_clip):
    normalized = robust_mri_to_unit(image_slice, z_clip)
    resized = resize_numpy(
        normalized, (img_size, img_size), order=3
    ).astype(np.float32)
    three_channel = np.repeat(resized[None, :, :], 3, axis=0)
    return ((three_channel - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def dice_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    if medpy_dc is not None:
        return float(medpy_dc(prediction, target))
    intersection = np.logical_and(prediction, target).sum()
    return 2.0 * float(intersection) / float(
        prediction.sum() + target.sum()
    )


def scipy_hd95(prediction, target):
    pred_surface = np.logical_xor(
        prediction, binary_erosion(prediction)
    )
    target_surface = np.logical_xor(target, binary_erosion(target))
    if not pred_surface.any() or not target_surface.any():
        return 0.0
    target_distance = distance_transform_edt(~target_surface)
    pred_distance = distance_transform_edt(~pred_surface)
    distances = np.concatenate([
        target_distance[pred_surface], pred_distance[target_surface]
    ])
    return float(np.percentile(distances, 95))


def hd95_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return 100.0
    if medpy_hd95 is not None:
        try:
            return float(medpy_hd95(prediction, target))
        except Exception:
            pass
    return scipy_hd95(prediction, target)


@torch.no_grad()
def infer_volume(model, image, img_size, inference_batch_size, device,
                 channels_last, z_clip):
    prediction = np.zeros_like(image, dtype=np.uint8)
    depth = image.shape[0]
    for start in range(0, depth, inference_batch_size):
        end = min(start + inference_batch_size, depth)
        batch_np = np.stack([
            prepare_slice(image[index], img_size, z_clip)
            for index in range(start, end)
        ], axis=0)
        batch = torch.from_numpy(batch_np).to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            batch = batch.contiguous(memory_format=torch.channels_last)
        logits = model(batch)["out"]
        masks = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        for offset, index in enumerate(range(start, end)):
            prediction[index] = resize_numpy(
                masks[offset], image[index].shape, order=0
            ).astype(np.uint8)
    return prediction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/roy/data/ACDC")
    parser.add_argument(
        "--list_dir", default="/home/roy/data/ACDC/lists_ACDC"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--checkpoint",
        default="./model/MCSF_ResUNet_ACDC_bs12_150e/best_model.pth"
    )
    parser.add_argument(
        "--output_dir",
        default="./test_log/MCSF_ResUNet_ACDC_bs12_150e_best"
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--z_clip", type=float, default=3.0)
    parser.add_argument("--channels_last", type=str2bool, default=True)
    parser.add_argument("--save_npz", type=str2bool, default=False)
    return parser.parse_args()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    logger.info("Arguments: %s", json.dumps(vars(args), indent=2))

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MCSFResUNet(
        num_classes=args.num_classes, pretrained_path=None
    ).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(
        "Loaded checkpoint=%s, epoch=%s, best_val_dice=%s",
        args.checkpoint,
        checkpoint.get("epoch") if isinstance(checkpoint, dict) else "unknown",
        checkpoint.get("best_val_dice") if isinstance(checkpoint, dict) else "unknown",
    )

    names = read_split_names(args.data_root, args.list_dir, args.split)
    paths = [
        find_sample_path(args.data_root, args.split, name)
        for name in names
    ]
    logger.info("Test cases: %d", len(paths))

    all_metrics = []
    case_results = {}
    for index, path in enumerate(paths):
        image, label = load_image_label(path)
        prediction = infer_volume(
            model=model,
            image=image,
            img_size=args.img_size,
            inference_batch_size=args.inference_batch_size,
            device=device,
            channels_last=args.channels_last,
            z_clip=args.z_clip,
        )
        metrics = []
        for class_index in range(1, args.num_classes):
            metrics.append([
                dice_score(prediction == class_index, label == class_index),
                hd95_score(prediction == class_index, label == class_index),
            ])
        metrics = np.asarray(metrics, dtype=np.float32)
        all_metrics.append(metrics)
        case_name = path.stem
        mean_dice = float(metrics[:, 0].mean())
        mean_hd95 = float(metrics[:, 1].mean())
        case_results[case_name] = {
            "mean_dice": mean_dice,
            "mean_hd95": mean_hd95,
        }
        logger.info(
            "idx %d case %s mean_dice %.6f mean_hd95 %.6f",
            index, case_name, mean_dice, mean_hd95
        )
        if args.save_npz:
            np.savez_compressed(
                str(output_dir / "{}_pred.npz".format(case_name)),
                pred=prediction, label=label
            )

    stacked = np.stack(all_metrics, axis=0)
    mean_per_class = np.mean(stacked, axis=0)
    summary = {"classes": {}, "cases": case_results}
    for class_index in range(1, args.num_classes):
        class_name = (
            CLASS_NAMES[class_index]
            if class_index < len(CLASS_NAMES)
            else "Class {}".format(class_index)
        )
        dice = float(mean_per_class[class_index - 1, 0])
        hd95 = float(mean_per_class[class_index - 1, 1])
        logger.info(
            "Mean class %d (%s): mean_dice %.6f mean_hd95 %.6f",
            class_index, class_name, dice, hd95
        )
        summary["classes"][str(class_index)] = {
            "name": class_name, "dice": dice, "hd95": hd95
        }

    summary["mean_dice"] = float(mean_per_class[:, 0].mean())
    summary["mean_hd95"] = float(mean_per_class[:, 1].mean())
    logger.info(
        "Testing performance: mean_dice : %.6f mean_hd95 : %.6f",
        summary["mean_dice"], summary["mean_hd95"]
    )
    with open(str(output_dir / "metrics.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main(parse_args())
