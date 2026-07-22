# -*- coding: utf-8 -*-
"""Test MCSF-ResUNet on Synapse test_vol_h5 without saving predictions."""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, zoom

from networks.mcsf_resunet import MCSFResUNet


CLASS_NAMES = [
    "Background", "Aorta", "Gallbladder", "Left kidney", "Right kidney",
    "Liver", "Pancreas", "Spleen", "Stomach"
]
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(value)


def setup_logger(output_dir):
    logger = logging.getLogger("mcsf_test")
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


def dice_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    intersection = np.logical_and(prediction, target).sum()
    return 2.0 * float(intersection) / float(prediction.sum() + target.sum())


def hd95_score(prediction, target):
    # Kept identical to the user's established protocol: pixel distance, no
    # voxel spacing, and max(shape) for one-empty cases.
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float(max(prediction.shape))
    pred_surface = np.logical_xor(prediction, binary_erosion(prediction))
    target_surface = np.logical_xor(target, binary_erosion(target))
    if not pred_surface.any() or not target_surface.any():
        return 0.0
    target_distance = distance_transform_edt(~target_surface)
    pred_distance = distance_transform_edt(~pred_surface)
    distances = np.concatenate([
        target_distance[pred_surface], pred_distance[target_surface]
    ])
    return float(np.percentile(distances, 95))


def resize2d(array, size, order):
    if tuple(array.shape) == tuple(size):
        return array
    return zoom(
        array,
        (float(size[0]) / array.shape[0], float(size[1]) / array.shape[1]),
        order=order,
    )


def volume_path(data_root, name):
    directory = Path(data_root) / "test_vol_h5"
    candidates = [
        directory / name,
        directory / "{}.h5".format(name),
        directory / "{}.npy.h5".format(name),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Test volume not found: {}".format(name))


def prepare_slice(image_slice, img_size):
    resized = resize2d(image_slice, (img_size, img_size), order=3).astype(np.float32)
    three_channel = np.repeat(resized[None, :, :], 3, axis=0)
    normalized = (three_channel - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.astype(np.float32)


@torch.no_grad()
def infer_volume(model, image, img_size, inference_batch_size, device, channels_last):
    prediction = np.zeros_like(image, dtype=np.uint8)
    depth = image.shape[0]
    for start in range(0, depth, inference_batch_size):
        end = min(start + inference_batch_size, depth)
        batch_np = np.stack([
            prepare_slice(image[index], img_size)
            for index in range(start, end)
        ], axis=0)
        batch = torch.from_numpy(batch_np).to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            batch = batch.contiguous(memory_format=torch.channels_last)
        logits = model(batch)["out"]
        masks = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        for offset, index in enumerate(range(start, end)):
            original_shape = image[index].shape
            prediction[index] = resize2d(
                masks[offset], original_shape, order=0
            ).astype(np.uint8)
    return prediction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/roy/data/Synapse")
    parser.add_argument("--list_dir", default="./lists/lists_Synapse")
    parser.add_argument(
        "--checkpoint", default="./model/MCSF_ResUNet_Synapse/last_model.pth"
    )
    parser.add_argument(
        "--output_dir", default="./test_log/MCSF_ResUNet_Synapse"
    )
    parser.add_argument("--split", default="test_vol")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--channels_last", type=str2bool, default=True)
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
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(
        "Loaded checkpoint=%s, epoch=%s, state=model",
        args.checkpoint,
        checkpoint.get("epoch") if isinstance(checkpoint, dict) else "unknown",
    )

    list_path = Path(args.list_dir) / "{}.txt".format(args.split)
    cases = [
        line.strip() for line in list_path.read_text().splitlines()
        if line.strip()
    ]

    dices = [[] for _ in range(args.num_classes)]
    hd95s = [[] for _ in range(args.num_classes)]
    case_results = {}

    for index, case_name in enumerate(cases):
        with h5py.File(str(volume_path(args.data_root, case_name)), "r") as handle:
            image = handle["image"][:].astype(np.float32)
            label = handle["label"][:].astype(np.uint8)

        prediction = infer_volume(
            model=model,
            image=image,
            img_size=args.img_size,
            inference_batch_size=args.inference_batch_size,
            device=device,
            channels_last=args.channels_last,
        )

        case_dices = []
        case_hd95s = []
        for class_index in range(1, args.num_classes):
            dice = dice_score(prediction == class_index, label == class_index)
            hd95 = hd95_score(prediction == class_index, label == class_index)
            dices[class_index].append(dice)
            hd95s[class_index].append(hd95)
            case_dices.append(dice)
            case_hd95s.append(hd95)

        mean_dice = float(np.mean(case_dices))
        mean_hd95 = float(np.mean(case_hd95s))
        case_results[case_name] = {
            "mean_dice": mean_dice,
            "mean_hd95": mean_hd95,
        }
        logger.info(
            "idx %d case %s mean_dice %.6f mean_hd95 %.6f",
            index, case_name, mean_dice, mean_hd95
        )

    summary = {"classes": {}, "cases": case_results}
    all_dices = []
    all_hd95s = []
    for class_index in range(1, args.num_classes):
        dice = float(np.mean(dices[class_index]))
        hd95 = float(np.mean(hd95s[class_index]))
        all_dices.append(dice)
        all_hd95s.append(hd95)
        name = (
            CLASS_NAMES[class_index]
            if class_index < len(CLASS_NAMES)
            else "Class {}".format(class_index)
        )
        logger.info(
            "Mean class %d (%s): mean_dice %.6f mean_hd95 %.6f",
            class_index, name, dice, hd95
        )
        summary["classes"][str(class_index)] = {
            "name": name, "dice": dice, "hd95": hd95
        }

    summary["mean_dice"] = float(np.mean(all_dices))
    summary["mean_hd95"] = float(np.mean(all_hd95s))
    logger.info(
        "Testing performance: mean_dice : %.6f mean_hd95 : %.6f",
        summary["mean_dice"], summary["mean_hd95"]
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main(parse_args())
