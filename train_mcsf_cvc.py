# -*- coding: utf-8 -*-
"""Train MCSF-ResUNet on CVC-ClinicDB binary polyp segmentation.

Default protocol:
- 550 training images from a fixed split file
- input size: 256x256
- batch size: 24
- 150 epochs
- two-class output: background/polyp
- 0.4 cross-entropy + 0.6 foreground Dice
- deep supervision: main + 0.4 aux1 + 0.2 aux2, normalized by 1.6
- no use of the 62-image test set during training

Compatible with Python 3.7, PyTorch 1.13.1 and torchvision 0.14.1.
"""

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from cvc_dataset import CVCClinicDBTrainDataset
from networks.mcsf_resunet import MCSFResUNet


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        "Invalid boolean value: {}".format(value)
    )


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
    logger = logging.getLogger("mcsf_cvc_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        str(output_dir / "train.log"), mode="a"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class ForegroundDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(ForegroundDiceLoss, self).__init__()
        self.smooth = float(smooth)

    def forward(self, logits, target):
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        foreground = (target == 1).float()
        dims = (0, 1, 2)
        intersection = torch.sum(probabilities * foreground, dim=dims)
        denominator = (
            torch.sum(probabilities, dim=dims)
            + torch.sum(foreground, dim=dims)
        )
        dice = (
            2.0 * intersection + self.smooth
        ) / (denominator + self.smooth)
        return 1.0 - dice


class StableBinarySegmentationLoss(nn.Module):
    def __init__(self, ce_weight=0.4, dice_weight=0.6):
        super(StableBinarySegmentationLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = ForegroundDiceLoss()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits, target):
        ce_value = self.ce(logits, target)
        dice_value = self.dice(logits, target)
        total = (
            self.ce_weight * ce_value
            + self.dice_weight * dice_value
        )
        return total, ce_value.detach(), dice_value.detach()


def set_learning_rates(
    optimizer,
    progress,
    warmup_epochs,
    max_epochs,
):
    if progress < warmup_epochs:
        schedule_factor = (
            0.1
            + 0.9
            * progress
            / max(float(warmup_epochs), 1.0)
        )
        warmup = True
    else:
        denominator = max(
            float(max_epochs - warmup_epochs), 1.0
        )
        cosine_progress = min(
            max(
                (progress - warmup_epochs) / denominator,
                0.0,
            ),
            1.0,
        )
        schedule_factor = 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
        warmup = False

    current_lrs = []
    for group in optimizer.param_groups:
        base_lr = float(group["base_lr"])
        min_lr = float(group["min_lr"])
        if warmup:
            lr = base_lr * schedule_factor
        else:
            lr = (
                min_lr
                + (base_lr - min_lr) * schedule_factor
            )
        group["lr"] = lr
        current_lrs.append(lr)
    return current_lrs


def clean_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    best_train_loss,
    args,
):
    checkpoint = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_train_loss": float(best_train_loss),
        "args": vars(args),
    }
    torch.save(checkpoint, str(path))


def load_resume(path, model, optimizer, scaler, logger):
    checkpoint = torch.load(path, map_location="cpu")
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(clean_state_dict(state), strict=True)

    start_epoch = 0
    best_train_loss = float("inf")
    if isinstance(checkpoint, dict):
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_train_loss = float(
            checkpoint.get("best_train_loss", float("inf"))
        )

    logger.info(
        "Resumed from %s, start_epoch=%d, best_train_loss=%.6f",
        path,
        start_epoch,
        best_train_loss,
    )
    return start_epoch, best_train_loss


def append_history(history_path, row):
    write_header = not history_path.is_file()
    with open(str(history_path), "a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "loss",
                "main_ce",
                "main_dice_loss",
                "train_dice",
                "encoder_lr",
                "new_lr",
                "seconds",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="/home/roy/data/CVC-ClinicDB",
    )
    parser.add_argument(
        "--train_list",
        default="/home/roy/data/CVC-ClinicDB/splits/train.txt",
    )
    parser.add_argument(
        "--pretrained_path",
        default=(
            "/home/roy/transunet/pretrained_weights/"
            "resnet34-b627a593.pth"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "./model/"
            "MCSF_ResUNet_CVC_ClinicDB_bs24_256_150e"
        ),
    )
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--new_lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument(
        "--channels_last", type=str2bool, default=True
    )
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def main(args):
    if args.num_classes != 2:
        raise ValueError(
            "CVC-ClinicDB is configured for binary segmentation; "
            "--num_classes must be 2."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    logger.info(
        "Arguments:\n%s",
        json.dumps(vars(args), indent=2),
    )

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    dataset = CVCClinicDBTrainDataset(
        data_root=args.data_root,
        split_file=args.train_list,
        img_size=args.img_size,
        augment=True,
    )
    if len(dataset) != 550:
        raise RuntimeError(
            "Expected 550 training images, found {}.".format(
                len(dataset)
            )
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(args.num_workers > 0),
    )
    logger.info(
        "Training samples=%d, batches/epoch=%d",
        len(dataset),
        len(loader),
    )

    model = MCSFResUNet(
        num_classes=args.num_classes,
        pretrained_path=(
            args.pretrained_path if not args.resume else None
        ),
    ).to(device)

    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if not args.resume:
        info = model.pretrained_info
        logger.info(
            "ResNet-34 pretrained loading: loaded=%d, "
            "missing=%s, unexpected=%s",
            info["loaded"],
            info["missing"],
            info["unexpected"],
        )

    encoder_params = list(model.encoder_parameters())
    new_params = list(model.new_parameters())
    encoder_min_lr = (
        args.encoder_lr * (args.min_lr / args.new_lr)
    )
    optimizer = torch.optim.AdamW(
        [
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
        ],
        weight_decay=args.weight_decay,
    )

    criterion = StableBinarySegmentationLoss(
        ce_weight=0.4,
        dice_weight=0.6,
    )
    scaler = GradScaler(
        enabled=(args.amp and device.type == "cuda")
    )

    start_epoch = 0
    best_train_loss = float("inf")
    if args.resume:
        start_epoch, best_train_loss = load_resume(
            args.resume,
            model,
            optimizer,
            scaler,
            logger,
        )

    history_path = output_dir / "history.csv"
    total_start = time.time()

    for epoch in range(start_epoch, args.max_epochs):
        epoch_number = epoch + 1
        epoch_start = time.time()

        shallow_frozen = epoch < args.freeze_epochs
        model.set_shallow_encoder_trainable(
            not shallow_frozen
        )
        model.train()
        if shallow_frozen:
            model.keep_frozen_shallow_eval()

        running_loss = 0.0
        running_ce = 0.0
        running_dice_loss = 0.0
        total_intersection = 0.0
        total_pred = 0.0
        total_gt = 0.0
        last_lrs = [args.encoder_lr, args.new_lr]

        for step, batch in enumerate(loader):
            progress = (
                epoch
                + float(step) / max(float(len(loader)), 1.0)
            )
            last_lrs = set_learning_rates(
                optimizer,
                progress,
                args.warmup_epochs,
                args.max_epochs,
            )

            images = batch["image"].to(
                device, non_blocking=True
            )
            labels = batch["label"].to(
                device, non_blocking=True
            )

            if args.channels_last and device.type == "cuda":
                images = images.contiguous(
                    memory_format=torch.channels_last
                )

            optimizer.zero_grad(set_to_none=True)
            with autocast(
                enabled=(args.amp and device.type == "cuda")
            ):
                outputs = model(images)
                main_loss, main_ce, main_dice = criterion(
                    outputs["out"], labels
                )
                aux1_loss, _, _ = criterion(
                    outputs["aux1"], labels
                )
                aux2_loss, _, _ = criterion(
                    outputs["aux2"], labels
                )
                loss = (
                    main_loss
                    + 0.4 * aux1_loss
                    + 0.2 * aux2_loss
                ) / 1.6

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().item())
            running_ce += float(main_ce.item())
            running_dice_loss += float(main_dice.item())

            with torch.no_grad():
                prediction = torch.argmax(
                    outputs["out"], dim=1
                )
                pred_fg = prediction == 1
                gt_fg = labels == 1
                total_intersection += float(
                    torch.logical_and(
                        pred_fg, gt_fg
                    ).sum().item()
                )
                total_pred += float(pred_fg.sum().item())
                total_gt += float(gt_fg.sum().item())

        epoch_loss = running_loss / max(len(loader), 1)
        epoch_ce = running_ce / max(len(loader), 1)
        epoch_dice_loss = (
            running_dice_loss / max(len(loader), 1)
        )
        train_dice = (
            2.0 * total_intersection + 1e-7
        ) / (total_pred + total_gt + 1e-7)
        epoch_seconds = time.time() - epoch_start

        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            save_checkpoint(
                output_dir / "best_train_loss.pth",
                model,
                optimizer,
                scaler,
                epoch_number,
                best_train_loss,
                args,
            )

        save_checkpoint(
            output_dir / "last_model.pth",
            model,
            optimizer,
            scaler,
            epoch_number,
            best_train_loss,
            args,
        )

        if (
            args.save_interval > 0
            and epoch_number % args.save_interval == 0
        ):
            save_checkpoint(
                output_dir
                / "epoch_{:03d}.pth".format(epoch_number),
                model,
                optimizer,
                scaler,
                epoch_number,
                best_train_loss,
                args,
            )

        append_history(
            history_path,
            {
                "epoch": epoch_number,
                "loss": "{:.8f}".format(epoch_loss),
                "main_ce": "{:.8f}".format(epoch_ce),
                "main_dice_loss": "{:.8f}".format(
                    epoch_dice_loss
                ),
                "train_dice": "{:.8f}".format(
                    train_dice
                ),
                "encoder_lr": "{:.10f}".format(
                    last_lrs[0]
                ),
                "new_lr": "{:.10f}".format(
                    last_lrs[1]
                ),
                "seconds": "{:.2f}".format(
                    epoch_seconds
                ),
            },
        )

        logger.info(
            "Epoch %03d/%03d | loss=%.6f | "
            "main_ce=%.6f | main_dice_loss=%.6f | "
            "train_dice=%.6f | lr=(%.2e, %.2e) | "
            "frozen=%s | %.1fs",
            epoch_number,
            args.max_epochs,
            epoch_loss,
            epoch_ce,
            epoch_dice_loss,
            train_dice,
            last_lrs[0],
            last_lrs[1],
            shallow_frozen,
            epoch_seconds,
        )

    logger.info(
        "Training finished in %.2f hours. "
        "Official testing should use last_model.pth, "
        "because no validation set was used.",
        (time.time() - total_start) / 3600.0,
    )


if __name__ == "__main__":
    main(parse_args())
