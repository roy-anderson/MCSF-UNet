# -*- coding: utf-8 -*-
"""Train MCSF-ResUNet on the 2D ACDC cardiac MRI dataset.

Expected layout:
  /home/roy/data/ACDC/
    train/
    valid/            # or val/
    test/
    lists_ACDC/
      train.txt
      valid.txt       # or val.txt
      test.txt

Supported sample files: .h5, .npz and dict-like .npy containing image/label.
A listed 3D training volume is automatically expanded into 2D slices.

Compatible with Python 3.7, PyTorch 1.13.1 and torchvision 0.14.1.
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt, zoom
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from networks.mcsf_resunet import MCSFResUNet

try:
    from medpy.metric.binary import dc as medpy_dc
    from medpy.metric.binary import hd95 as medpy_hd95
except Exception:
    medpy_dc = None
    medpy_hd95 = None


CLASS_NAMES = ["Background", "Right ventricle", "Myocardium", "Left ventricle"]
IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(3, 1, 1)
IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(3, 1, 1)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Invalid boolean value: {}".format(value))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logger(output_dir):
    logger = logging.getLogger("mcsf_acdc_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(output_dir / "train.log", mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def split_directory(data_root, split):
    root = Path(data_root)
    aliases = {
        "train": ["train", "training"],
        "valid": ["valid", "val", "validation"],
        "val": ["val", "valid", "validation"],
        "test": ["test", "testing"],
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
            names = [line.strip() for line in handle if line.strip()]
        return names

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
    suffixes = ["", ".h5", ".npz", ".npy", ".npy.h5"]
    for base in bases:
        if base.is_file():
            return base
        for suffix in suffixes:
            candidate = Path(str(base) + suffix)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "Cannot find sample '{}' for split '{}' under {}".format(
            name, split, data_root
        )
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
            raise ValueError(
                "Plain .npy is unsupported because image and label must be stored together: {}".format(path)
            )
        obj = array.item()
        image = obj.get("image", obj.get("img", obj.get("data")))
        label = obj.get("label", obj.get("mask", obj.get("seg")))
        if image is None or label is None:
            raise KeyError(
                "Cannot locate image/label in dict-like file {}".format(path)
            )
    else:
        raise ValueError("Unsupported file type: {}".format(path))

    image = np.asarray(image)
    label = np.asarray(label)
    if image.shape != label.shape:
        raise ValueError(
            "Image shape {} differs from label shape {} in {}".format(
                image.shape, label.shape, path
            )
        )
    return image, label


def image_shape(path):
    if path.suffix.lower() == ".h5":
        with h5py.File(str(path), "r") as handle:
            keys = list(handle.keys())
            image_key = choose_key(keys, ["image", "img", "data"])
            if image_key is None:
                raise KeyError("Cannot find image key in {}".format(path))
            return tuple(handle[image_key].shape)
    image, _ = load_image_label(path)
    return tuple(image.shape)


def resize_numpy(array, output_size, order):
    if tuple(array.shape[-2:]) == tuple(output_size):
        return array
    factors = (
        float(output_size[0]) / float(array.shape[-2]),
        float(output_size[1]) / float(array.shape[-1]),
    )
    return zoom(array, factors, order=order)


def robust_mri_to_unit(image, z_clip=3.0):
    """Per-slice z-score, robust clipping, then mapping to [0, 1]."""
    image = image.astype(np.float32)
    mean = float(image.mean())
    std = float(image.std())
    if std > 1e-6:
        image = (image - mean) / std
    else:
        image = image - mean
    z_clip = max(float(z_clip), 1e-3)
    image = np.clip(image, -z_clip, z_clip)
    image = (image + z_clip) / (2.0 * z_clip)
    return image.astype(np.float32)


class ACDCSliceDataset(Dataset):
    def __init__(self, data_root, list_dir, split="train", img_size=224,
                 augment=True, z_clip=3.0):
        self.data_root = data_root
        self.list_dir = list_dir
        self.split = split
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.z_clip = float(z_clip)

        names = read_split_names(data_root, list_dir, split)
        self.items = []
        for name in names:
            path = find_sample_path(data_root, split, name)
            shape = image_shape(path)
            if len(shape) == 2:
                self.items.append((path, None))
            elif len(shape) == 3:
                for slice_index in range(shape[0]):
                    self.items.append((path, slice_index))
            else:
                raise ValueError(
                    "Expected 2D or 3D sample, got {} in {}".format(shape, path)
                )
        if not self.items:
            raise RuntimeError("No training samples were found.")

    def __len__(self):
        return len(self.items)

    def _augment(self, image, label):
        if random.random() < 0.7:
            angle = random.uniform(-15.0, 15.0)
            scale = random.uniform(0.90, 1.10)
            image = TF.affine(
                image, angle=angle, translate=[0, 0], scale=scale,
                shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR,
                fill=0.0
            )
            label_float = TF.affine(
                label.float(), angle=angle, translate=[0, 0], scale=scale,
                shear=[0.0, 0.0], interpolation=InterpolationMode.NEAREST,
                fill=0.0
            )
            label = label_float.long()

        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
        if random.random() < 0.2:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[1])

        if random.random() < 0.5:
            contrast = random.uniform(0.90, 1.10)
            brightness = random.uniform(-0.05, 0.05)
            image = torch.clamp(image * contrast + brightness, 0.0, 1.0)
        return image, label

    def __getitem__(self, index):
        path, slice_index = self.items[index]
        image, label = load_image_label(path)
        if image.ndim == 3:
            image = image[slice_index]
            label = label[slice_index]

        image = robust_mri_to_unit(image, self.z_clip)
        image = resize_numpy(
            image, (self.img_size, self.img_size), order=3
        ).astype(np.float32)
        label = resize_numpy(
            label, (self.img_size, self.img_size), order=0
        )
        label = np.rint(label).astype(np.int64)

        image_t = torch.from_numpy(image).float().unsqueeze(0)
        label_t = torch.from_numpy(label).long().unsqueeze(0)
        if self.augment:
            image_t, label_t = self._augment(image_t, label_t)

        image_t = image_t.repeat(3, 1, 1)
        image_t = (image_t - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": image_t,
            "label": label_t.squeeze(0),
            "case_name": path.stem,
        }


class ACDCVolumeDataset(Dataset):
    def __init__(self, data_root, list_dir, split="valid"):
        names = read_split_names(data_root, list_dir, split)
        self.paths = [
            find_sample_path(data_root, split, name) for name in names
        ]
        if not self.paths:
            raise RuntimeError(
                "No volume samples found for split '{}'".format(split)
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image, label = load_image_label(path)
        if image.ndim == 2:
            image = image[None, ...]
            label = label[None, ...]
        if image.ndim != 3:
            raise ValueError(
                "Expected 3D volume or 2D slice, got {} in {}".format(
                    image.shape, path
                )
            )
        return {
            "image": image.astype(np.float32),
            "label": label.astype(np.uint8),
            "case_name": path.stem,
        }


class ForegroundDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        super(ForegroundDiceLoss, self).__init__()
        self.num_classes = int(num_classes)
        self.smooth = float(smooth)

    def forward(self, logits, target):
        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(
            target.long(), num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()
        probabilities = probabilities[:, 1:]
        target_one_hot = target_one_hot[:, 1:]
        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * target_one_hot, dim=dims)
        denominator = (
            torch.sum(probabilities, dim=dims) +
            torch.sum(target_one_hot, dim=dims)
        )
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        return 1.0 - dice.mean()


class StableSegmentationLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.4, dice_weight=0.6):
        super(StableSegmentationLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = ForegroundDiceLoss(num_classes)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits, target):
        ce_value = self.ce(logits, target)
        dice_value = self.dice(logits, target)
        total = self.ce_weight * ce_value + self.dice_weight * dice_value
        return total, ce_value.detach(), dice_value.detach()


def set_learning_rates(optimizer, progress, warmup_epochs, max_epochs):
    if progress < warmup_epochs:
        factor = 0.1 + 0.9 * progress / max(float(warmup_epochs), 1.0)
        schedule_factor = factor
    else:
        denominator = max(float(max_epochs - warmup_epochs), 1.0)
        cosine_progress = min(
            max((progress - warmup_epochs) / denominator, 0.0), 1.0
        )
        schedule_factor = 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )

    current_lrs = []
    for group in optimizer.param_groups:
        base_lr = group["base_lr"]
        min_lr = group["min_lr"]
        if progress < warmup_epochs:
            lr = base_lr * schedule_factor
        else:
            lr = min_lr + (base_lr - min_lr) * schedule_factor
        group["lr"] = lr
        current_lrs.append(lr)
    return current_lrs


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


def prepare_slice(image_slice, img_size, z_clip):
    normalized = robust_mri_to_unit(image_slice, z_clip)
    resized = resize_numpy(
        normalized, (img_size, img_size), order=3
    ).astype(np.float32)
    three_channel = np.repeat(resized[None, :, :], 3, axis=0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return ((three_channel - mean) / std).astype(np.float32)


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
            original_shape = image[index].shape
            prediction[index] = resize_numpy(
                masks[offset], original_shape, order=0
            ).astype(np.uint8)
    return prediction


@torch.no_grad()
def evaluate_model(model, dataset, num_classes, img_size,
                   inference_batch_size, device, channels_last, z_clip,
                   logger=None, prefix="Validation"):
    model.eval()
    all_metrics = []
    for index in range(len(dataset)):
        sample = dataset[index]
        prediction = infer_volume(
            model=model,
            image=sample["image"],
            img_size=img_size,
            inference_batch_size=inference_batch_size,
            device=device,
            channels_last=channels_last,
            z_clip=z_clip,
        )
        metrics = []
        for class_index in range(1, num_classes):
            metrics.append([
                dice_score(
                    prediction == class_index,
                    sample["label"] == class_index
                ),
                hd95_score(
                    prediction == class_index,
                    sample["label"] == class_index
                ),
            ])
        metrics = np.asarray(metrics, dtype=np.float32)
        all_metrics.append(metrics)
        if logger is not None:
            logger.info(
                "%s idx %d case %s mean_dice %.6f mean_hd95 %.6f",
                prefix, index, sample["case_name"],
                float(metrics[:, 0].mean()),
                float(metrics[:, 1].mean()),
            )
    stacked = np.stack(all_metrics, axis=0)
    mean_per_class = np.mean(stacked, axis=0)
    return {
        "mean_dice": float(mean_per_class[:, 0].mean()),
        "mean_hd95": float(mean_per_class[:, 1].mean()),
        "per_class": mean_per_class,
    }


def save_checkpoint(path, model, optimizer, scaler, epoch, args,
                    best_val_dice):
    checkpoint = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_dice": float(best_val_dice),
        "args": vars(args),
    }
    torch.save(checkpoint, str(path))


def load_resume(path, model, optimizer, scaler, logger):
    checkpoint = torch.load(path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError("Resume checkpoint does not contain key 'model'.")
    model.load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint.get("epoch", 0))
    best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
    logger.info(
        "Resumed from %s at completed epoch %d, best_val_dice %.6f",
        path, start_epoch, best_val_dice
    )
    return start_epoch, best_val_dice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/roy/data/ACDC")
    parser.add_argument(
        "--list_dir", default="/home/roy/data/ACDC/lists_ACDC"
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--valid_split", default="valid")
    parser.add_argument(
        "--output_dir", default="./model/MCSF_ResUNet_ACDC_bs12_150e"
    )
    parser.add_argument(
        "--pretrained_path",
        default="./pretrained_weights/resnet34-b627a593.pth"
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--new_lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument("--use_validation", type=str2bool, default=True)
    parser.add_argument("--val_start_epoch", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--z_clip", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--augment", type=str2bool, default=True)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--channels_last", type=str2bool, default=True)
    return parser.parse_args()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    logger.info("Arguments: %s", json.dumps(vars(args), indent=2))

    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_dataset = ACDCSliceDataset(
        data_root=args.data_root,
        list_dir=args.list_dir,
        split=args.train_split,
        img_size=args.img_size,
        augment=args.augment,
        z_clip=args.z_clip,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(**loader_kwargs)
    logger.info(
        "Training slices: %d; iterations/epoch: %d",
        len(train_dataset), len(train_loader)
    )

    valid_dataset = None
    if args.use_validation:
        try:
            valid_dataset = ACDCVolumeDataset(
                args.data_root, args.list_dir, args.valid_split
            )
            logger.info("Validation cases: %d", len(valid_dataset))
        except Exception as error:
            logger.warning(
                "Validation disabled because the valid split could not be loaded: %s",
                error
            )
            valid_dataset = None

    model = MCSFResUNet(
        num_classes=args.num_classes,
        pretrained_path=args.pretrained_path if not args.resume else None,
    ).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if not args.resume:
        info = model.pretrained_info
        logger.info(
            "ResNet-34 pretrained loading: loaded=%d, missing=%s, unexpected=%s",
            info["loaded"], info["missing"], info["unexpected"]
        )

    encoder_params = list(model.encoder_parameters())
    new_params = list(model.new_parameters())
    encoder_min_lr = args.encoder_lr * (args.min_lr / args.new_lr)
    optimizer = torch.optim.AdamW([
        {
            "params": encoder_params,
            "lr": args.encoder_lr,
            "base_lr": args.encoder_lr,
            "min_lr": encoder_min_lr,
            "name": "encoder",
        },
        {
            "params": new_params,
            "lr": args.new_lr,
            "base_lr": args.new_lr,
            "min_lr": args.min_lr,
            "name": "new_modules",
        },
    ], weight_decay=args.weight_decay)

    criterion = StableSegmentationLoss(
        num_classes=args.num_classes, ce_weight=0.4, dice_weight=0.6
    )
    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))

    start_epoch = 0
    best_val_dice = -1.0
    if args.resume:
        start_epoch, best_val_dice = load_resume(
            args.resume, model, optimizer, scaler, logger
        )

    train_start = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        epoch_number = epoch + 1
        shallow_frozen = epoch < args.freeze_epochs
        model.set_shallow_encoder_trainable(not shallow_frozen)
        model.train()
        if shallow_frozen:
            model.keep_frozen_shallow_eval()

        running_total = 0.0
        running_ce = 0.0
        running_dice = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            progress = epoch + float(step) / max(float(len(train_loader)), 1.0)
            current_lrs = set_learning_rates(
                optimizer, progress, args.warmup_epochs, args.max_epochs
            )

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if args.channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(args.amp and device.type == "cuda")):
                outputs = model(images)
                main_loss, main_ce, main_dice = criterion(
                    outputs["out"], labels
                )
                aux1_loss, _, _ = criterion(outputs["aux1"], labels)
                aux2_loss, _, _ = criterion(outputs["aux2"], labels)
                loss = (
                    main_loss + 0.4 * aux1_loss + 0.2 * aux2_loss
                ) / 1.6

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()

            running_total += float(loss.detach().item())
            running_ce += float(main_ce.item())
            running_dice += float(main_dice.item())

            if step == 0 or (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                logger.info(
                    "Epoch %03d/%03d step %03d/%03d loss %.5f main_ce %.5f "
                    "main_dice_loss %.5f lr_enc %.3e lr_new %.3e frozen=%s",
                    epoch_number, args.max_epochs, step + 1,
                    len(train_loader), running_total / float(step + 1),
                    running_ce / float(step + 1),
                    running_dice / float(step + 1),
                    current_lrs[0], current_lrs[1], shallow_frozen
                )

        epoch_seconds = time.time() - epoch_start
        elapsed = time.time() - train_start
        completed = epoch_number - start_epoch
        remaining_epochs = args.max_epochs - epoch_number
        estimated_remaining = (
            elapsed / max(float(completed), 1.0) * remaining_epochs
        )
        logger.info(
            "Epoch %03d completed: loss %.6f, time %.1fs, estimated remaining %.2fh",
            epoch_number,
            running_total / max(float(len(train_loader)), 1.0),
            epoch_seconds,
            estimated_remaining / 3600.0,
        )

        should_validate = (
            valid_dataset is not None and
            epoch_number >= args.val_start_epoch and
            epoch_number % args.val_interval == 0
        )
        if should_validate:
            result = evaluate_model(
                model=model,
                dataset=valid_dataset,
                num_classes=args.num_classes,
                img_size=args.img_size,
                inference_batch_size=args.inference_batch_size,
                device=device,
                channels_last=args.channels_last,
                z_clip=args.z_clip,
                logger=logger,
                prefix="Validation",
            )
            logger.info(
                "Validation epoch %03d: mean_dice %.6f mean_hd95 %.6f",
                epoch_number, result["mean_dice"], result["mean_hd95"]
            )
            for class_index in range(1, args.num_classes):
                class_name = (
                    CLASS_NAMES[class_index]
                    if class_index < len(CLASS_NAMES)
                    else "Class {}".format(class_index)
                )
                logger.info(
                    "Validation class %d (%s): dice %.6f hd95 %.6f",
                    class_index, class_name,
                    float(result["per_class"][class_index - 1, 0]),
                    float(result["per_class"][class_index - 1, 1]),
                )
            if result["mean_dice"] > best_val_dice:
                best_val_dice = result["mean_dice"]
                save_checkpoint(
                    output_dir / "best_model.pth", model, optimizer,
                    scaler, epoch_number, args, best_val_dice
                )
                logger.info(
                    "Saved new best_model.pth: epoch=%d, val_dice=%.6f",
                    epoch_number, best_val_dice
                )

        if epoch_number % args.save_interval == 0:
            checkpoint_path = output_dir / "epoch_{:03d}.pth".format(
                epoch_number
            )
            save_checkpoint(
                checkpoint_path, model, optimizer, scaler,
                epoch_number, args, best_val_dice
            )
            logger.info("Saved checkpoint: %s", checkpoint_path)

        save_checkpoint(
            output_dir / "last_model.pth", model, optimizer,
            scaler, epoch_number, args, best_val_dice
        )

    if valid_dataset is not None and not (output_dir / "best_model.pth").is_file():
        result = evaluate_model(
            model=model,
            dataset=valid_dataset,
            num_classes=args.num_classes,
            img_size=args.img_size,
            inference_batch_size=args.inference_batch_size,
            device=device,
            channels_last=args.channels_last,
            z_clip=args.z_clip,
            logger=logger,
            prefix="Final validation",
        )
        best_val_dice = result["mean_dice"]
        save_checkpoint(
            output_dir / "best_model.pth", model, optimizer,
            scaler, args.max_epochs, args, best_val_dice
        )

    total_hours = (time.time() - train_start) / 3600.0
    logger.info(
        "Training finished in %.3f hours; best_val_dice=%.6f",
        total_hours, best_val_dice
    )


if __name__ == "__main__":
    main(parse_args())
