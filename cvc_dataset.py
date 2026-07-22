# -*- coding: utf-8 -*-
"""CVC-ClinicDB dataset utilities for MCSF-ResUNet.

Expected layout:
    /home/roy/data/CVC-ClinicDB/
        Original/
            1.png
            ...
        Ground Truth/
            1.png
            ...
        splits/
            train.txt
            test.txt

Compatible with Python 3.7, PyTorch 1.13.1 and torchvision 0.14.1.
"""

import re
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(3, 1, 1)
IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(3, 1, 1)


def natural_key(value):
    """Natural sort key: 2.png comes before 10.png."""
    parts = re.split(r"(\d+)", str(value))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def build_file_index(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            "Directory does not exist: {}".format(directory)
        )

    index = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if stem in index:
            raise RuntimeError(
                "Duplicate stem '{}' in {}: {} and {}".format(
                    stem, directory, index[stem], path
                )
            )
        index[stem] = path
    return index


def read_split_file(split_file):
    split_file = Path(split_file)
    if not split_file.is_file():
        raise FileNotFoundError(
            "Split file does not exist: {}".format(split_file)
        )
    with open(str(split_file), "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if not names:
        raise RuntimeError("Split file is empty: {}".format(split_file))
    return names


def resolve_pairs(data_root, split_file):
    data_root = Path(data_root)
    image_index = build_file_index(data_root / "Original")
    mask_index = build_file_index(data_root / "Ground Truth")
    names = read_split_file(split_file)

    pairs = []
    missing = []
    for name in names:
        stem = Path(name).stem
        image_path = image_index.get(stem)
        mask_path = mask_index.get(stem)
        if image_path is None or mask_path is None:
            missing.append(
                (stem, image_path is not None, mask_path is not None)
            )
            continue
        pairs.append((stem, image_path, mask_path))

    if missing:
        preview = missing[:10]
        raise FileNotFoundError(
            "Some image-mask pairs are missing. "
            "Format=(stem, image_exists, mask_exists), examples={}".format(
                preview
            )
        )
    return pairs


def _resize_pair(image, mask, img_size):
    size = [int(img_size), int(img_size)]
    image = TF.resize(
        image, size, interpolation=InterpolationMode.BILINEAR,
        antialias=True
    )
    mask = TF.resize(
        mask, size, interpolation=InterpolationMode.NEAREST
    )
    return image, mask


def _apply_train_augmentation(image, mask):
    # Geometric transforms are applied identically to image and mask.
    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    if random.random() < 0.2:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    if random.random() < 0.7:
        angle = random.uniform(-15.0, 15.0)
        scale = random.uniform(0.90, 1.10)
        image = TF.affine(
            image,
            angle=angle,
            translate=[0, 0],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=[0, 0],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )

    # Appearance transforms are applied only to the RGB image.
    if random.random() < 0.5:
        image = TF.adjust_brightness(
            image, random.uniform(0.85, 1.15)
        )
        image = TF.adjust_contrast(
            image, random.uniform(0.85, 1.15)
        )
        image = TF.adjust_saturation(
            image, random.uniform(0.85, 1.15)
        )

    return image, mask


def image_to_tensor(image):
    image = TF.to_tensor(image)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image


def mask_to_tensor(mask):
    mask_np = np.asarray(mask, dtype=np.uint8)
    # Ground-truth masks are usually 0/255. Using >0 also supports 0/1 masks.
    mask_np = (mask_np > 0).astype(np.int64)
    return torch.from_numpy(mask_np).long()


class CVCClinicDBTrainDataset(Dataset):
    """Training dataset returning 256x256 RGB images and binary labels."""

    def __init__(
        self,
        data_root,
        split_file,
        img_size=256,
        augment=True,
    ):
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.pairs = resolve_pairs(self.data_root, self.split_file)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        name, image_path, mask_path = self.pairs[index]
        with Image.open(str(image_path)) as image_handle:
            image = image_handle.convert("RGB")
        with Image.open(str(mask_path)) as mask_handle:
            mask = mask_handle.convert("L")

        if image.size != mask.size:
            raise ValueError(
                "Image and mask sizes differ for '{}': {} vs {}".format(
                    name, image.size, mask.size
                )
            )

        image, mask = _resize_pair(image, mask, self.img_size)
        if self.augment:
            image, mask = _apply_train_augmentation(image, mask)

        return {
            "image": image_to_tensor(image),
            "label": mask_to_tensor(mask),
            "case_name": name,
        }


class CVCClinicDBTestDataset(Dataset):
    """Test dataset preserving original resolution for final metrics."""

    def __init__(self, data_root, split_file, img_size=256):
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.img_size = int(img_size)
        self.pairs = resolve_pairs(self.data_root, self.split_file)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        name, image_path, mask_path = self.pairs[index]
        with Image.open(str(image_path)) as image_handle:
            image = image_handle.convert("RGB")
        with Image.open(str(mask_path)) as mask_handle:
            mask = mask_handle.convert("L")

        if image.size != mask.size:
            raise ValueError(
                "Image and mask sizes differ for '{}': {} vs {}".format(
                    name, image.size, mask.size
                )
            )

        original_image = np.asarray(image, dtype=np.uint8)
        original_mask = (
            np.asarray(mask, dtype=np.uint8) > 0
        ).astype(np.uint8)

        resized_image, _ = _resize_pair(image, mask, self.img_size)
        return {
            "image": image_to_tensor(resized_image),
            "original_image": original_image,
            "original_mask": original_mask,
            "case_name": name,
            "original_size": (original_mask.shape[0], original_mask.shape[1]),
        }


def cvc_test_collate(batch):
    """Stack resized tensors while retaining variable-size originals."""
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "original_image": [item["original_image"] for item in batch],
        "original_mask": [item["original_mask"] for item in batch],
        "case_name": [item["case_name"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
    }
