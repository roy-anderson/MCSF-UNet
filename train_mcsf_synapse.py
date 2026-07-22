# -*- coding: utf-8 -*-
"""Train MCSF-ResUNet on the 2D Synapse slice dataset."""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from networks.mcsf_resunet import MCSFResUNet


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
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
    logger = logging.getLogger("mcsf_train")
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


def resize_numpy(array, output_size, order):
    if tuple(array.shape) == tuple(output_size):
        return array
    factors = (
        float(output_size[0]) / float(array.shape[0]),
        float(output_size[1]) / float(array.shape[1]),
    )
    return zoom(array, factors, order=order)


class SynapseSliceDataset(Dataset):
    def __init__(self, data_root, list_dir, split="train", img_size=224,
                 augment=True, swap_kidneys_on_flip=True):
        self.data_dir = Path(data_root) / "train_npz"
        list_path = Path(list_dir) / "{}.txt".format(split)
        if not list_path.is_file():
            raise FileNotFoundError("List file not found: {}".format(list_path))
        self.sample_names = [
            line.strip() for line in list_path.read_text().splitlines()
            if line.strip()
        ]
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.swap_kidneys_on_flip = bool(swap_kidneys_on_flip)

    def __len__(self):
        return len(self.sample_names)

    def _path(self, name):
        candidates = [
            self.data_dir / "{}.npz".format(name),
            self.data_dir / name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError("Training sample not found: {}".format(name))

    def _augment(self, image, label):
        # A single affine transform combines mild rotation and scale, reducing
        # CPU overhead compared with several independent resampling operations.
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
            if self.swap_kidneys_on_flip:
                # Class 3 = left kidney, class 4 = right kidney in the user's
                # established Synapse class mapping.
                old = label.clone()
                label[old == 3] = 4
                label[old == 4] = 3

        if random.random() < 0.5:
            contrast = random.uniform(0.90, 1.10)
            brightness = random.uniform(-0.05, 0.05)
            image = torch.clamp(image * contrast + brightness, 0.0, 1.0)
        return image, label

    def __getitem__(self, index):
        name = self.sample_names[index]
        data = np.load(str(self._path(name)))
        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.int64)

        image = resize_numpy(image, (self.img_size, self.img_size), order=3)
        label = resize_numpy(label, (self.img_size, self.img_size), order=0)
        label = np.rint(label).astype(np.int64)

        image_t = torch.from_numpy(image).float().unsqueeze(0)
        label_t = torch.from_numpy(label).long().unsqueeze(0)
        if self.augment:
            image_t, label_t = self._augment(image_t, label_t)

        image_t = image_t.repeat(3, 1, 1)
        image_t = (image_t - IMAGENET_MEAN) / IMAGENET_STD
        label_t = label_t.squeeze(0)
        return {"image": image_t, "label": label_t, "case_name": name}


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
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
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
        cosine_factor = factor
    else:
        denominator = max(float(max_epochs - warmup_epochs), 1.0)
        cosine_progress = min(max((progress - warmup_epochs) / denominator, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

    current_lrs = []
    for group in optimizer.param_groups:
        base_lr = group["base_lr"]
        min_lr = group["min_lr"]
        if progress < warmup_epochs:
            lr = base_lr * cosine_factor
        else:
            lr = min_lr + (base_lr - min_lr) * cosine_factor
        group["lr"] = lr
        current_lrs.append(lr)
    return current_lrs


def save_checkpoint(path, model, optimizer, scaler, epoch, args):
    checkpoint = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
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
    logger.info("Resumed from %s at completed epoch %d", path, start_epoch)
    return start_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/roy/data/Synapse")
    parser.add_argument("--list_dir", default="./lists/lists_Synapse")
    parser.add_argument("--output_dir", default="./model/MCSF_ResUNet_Synapse")
    parser.add_argument(
        "--pretrained_path",
        default="./pretrained_weights/resnet34-b627a593.pth"
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=9)
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--augment", type=str2bool, default=True)
    parser.add_argument("--swap_kidneys_on_flip", type=str2bool, default=True)
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

    dataset = SynapseSliceDataset(
        data_root=args.data_root,
        list_dir=args.list_dir,
        split="train",
        img_size=args.img_size,
        augment=args.augment,
        swap_kidneys_on_flip=args.swap_kidneys_on_flip,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = dict(
        dataset=dataset,
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
    loader = DataLoader(**loader_kwargs)
    logger.info("Training slices: %d; iterations/epoch: %d", len(dataset), len(loader))

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
    if args.resume:
        start_epoch = load_resume(
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

        for step, batch in enumerate(loader):
            progress = epoch + float(step) / max(float(len(loader)), 1.0)
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
                main_loss, main_ce, main_dice = criterion(outputs["out"], labels)
                aux1_loss, _, _ = criterion(outputs["aux1"], labels)
                aux2_loss, _, _ = criterion(outputs["aux2"], labels)
                loss = (main_loss + 0.4 * aux1_loss + 0.2 * aux2_loss) / 1.6

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_total += float(loss.detach().item())
            running_ce += float(main_ce.item())
            running_dice += float(main_dice.item())

            if step == 0 or (step + 1) % 25 == 0 or (step + 1) == len(loader):
                logger.info(
                    "Epoch %03d/%03d step %03d/%03d loss %.5f main_ce %.5f "
                    "main_dice_loss %.5f lr_enc %.3e lr_new %.3e frozen=%s",
                    epoch_number, args.max_epochs, step + 1, len(loader),
                    running_total / float(step + 1),
                    running_ce / float(step + 1),
                    running_dice / float(step + 1),
                    current_lrs[0], current_lrs[1], shallow_frozen
                )

        epoch_seconds = time.time() - epoch_start
        elapsed = time.time() - train_start
        completed = epoch_number - start_epoch
        remaining_epochs = args.max_epochs - epoch_number
        estimated_remaining = elapsed / max(float(completed), 1.0) * remaining_epochs
        logger.info(
            "Epoch %03d completed: loss %.6f, time %.1fs, estimated remaining %.2fh",
            epoch_number,
            running_total / max(float(len(loader)), 1.0),
            epoch_seconds,
            estimated_remaining / 3600.0,
        )

        if epoch_number % args.save_interval == 0:
            checkpoint_path = output_dir / "epoch_{:03d}.pth".format(epoch_number)
            save_checkpoint(
                checkpoint_path, model, optimizer, scaler, epoch_number, args
            )
            logger.info("Saved checkpoint: %s", checkpoint_path)

        save_checkpoint(
            output_dir / "last_model.pth",
            model, optimizer, scaler, epoch_number, args
        )

    total_hours = (time.time() - train_start) / 3600.0
    logger.info("Training finished in %.3f hours", total_hours)


if __name__ == "__main__":
    main(parse_args())
