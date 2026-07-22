# -*- coding: utf-8 -*-
"""Test MCSF-ResUNet on the 62-image CVC-ClinicDB test split.

Evaluation:
- resize RGB input to 256x256
- single forward pass, no TTA and no post-processing
- resize foreground probability back to original image size
- threshold at 0.5
- compute metrics per original-resolution image
- report mean and standard deviation

Metrics:
Dice, IoU, precision, recall, specificity, accuracy, MAE and HD95.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from cvc_dataset import (
    CVCClinicDBTestDataset,
    cvc_test_collate,
)
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


def setup_logger(output_dir):
    logger = logging.getLogger("mcsf_cvc_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        str(output_dir / "test.log"), mode="w"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def clean_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def resize_probability(probability, output_size):
    """Resize a float probability map to (height, width).

    PIL mode ``I;16`` does not support bilinear resizing in some Pillow
    versions. Mode ``F`` stores floating-point values and supports bilinear
    interpolation, so the probability map can be resized directly without
    16-bit quantization.
    """
    height, width = output_size
    probability = np.clip(
        probability.astype(np.float32), 0.0, 1.0
    )

    bilinear = (
        Image.Resampling.BILINEAR
        if hasattr(Image, "Resampling")
        else Image.BILINEAR
    )
    image = Image.fromarray(probability, mode="F")
    image = image.resize(
        (int(width), int(height)),
        resample=bilinear,
    )
    resized = np.asarray(image, dtype=np.float32)
    return np.clip(resized, 0.0, 1.0)


def hd95_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float(max(target.shape))

    pred_surface = np.logical_xor(
        prediction,
        binary_erosion(prediction),
    )
    target_surface = np.logical_xor(
        target,
        binary_erosion(target),
    )
    if not pred_surface.any() or not target_surface.any():
        return 0.0

    distance_to_target = distance_transform_edt(
        ~target_surface
    )
    distance_to_prediction = distance_transform_edt(
        ~pred_surface
    )
    distances = np.concatenate(
        [
            distance_to_target[pred_surface],
            distance_to_prediction[target_surface],
        ]
    )
    return float(np.percentile(distances, 95))


def calculate_metrics(prediction, target, probability):
    prediction = prediction.astype(np.uint8)
    target = target.astype(np.uint8)
    probability = probability.astype(np.float32)

    pred_bool = prediction.astype(bool)
    target_bool = target.astype(bool)

    tp = float(np.logical_and(pred_bool, target_bool).sum())
    fp = float(
        np.logical_and(pred_bool, ~target_bool).sum()
    )
    fn = float(
        np.logical_and(~pred_bool, target_bool).sum()
    )
    tn = float(
        np.logical_and(~pred_bool, ~target_bool).sum()
    )
    eps = 1e-7

    dice = (2.0 * tp + eps) / (
        2.0 * tp + fp + fn + eps
    )
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (
        tp + tn + fp + fn + eps
    )
    mae = float(
        np.mean(np.abs(probability - target.astype(np.float32)))
    )
    hd95 = hd95_score(prediction, target)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "mae": float(mae),
        "hd95": float(hd95),
    }


def save_prediction_outputs(
    output_dir,
    name,
    original_image,
    target,
    probability,
    prediction,
):
    mask_dir = output_dir / "pred_masks"
    probability_dir = output_dir / "probability_maps"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    probability_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(
        (prediction * 255).astype(np.uint8)
    ).save(str(mask_dir / "{}.png".format(name)))

    Image.fromarray(
        np.rint(probability * 255.0).astype(np.uint8)
    ).save(
        str(probability_dir / "{}.png".format(name))
    )

    # Visualization: overlap=yellow, GT-only=red, prediction-only=green.
    overlay = original_image.astype(np.float32).copy()
    gt = target.astype(bool)
    pred = prediction.astype(bool)
    overlap = np.logical_and(gt, pred)
    gt_only = np.logical_and(gt, ~pred)
    pred_only = np.logical_and(pred, ~gt)

    color = overlay.copy()
    color[overlap] = np.asarray(
        [255, 255, 0], dtype=np.float32
    )
    color[gt_only] = np.asarray(
        [255, 0, 0], dtype=np.float32
    )
    color[pred_only] = np.asarray(
        [0, 255, 0], dtype=np.float32
    )
    overlay = (
        0.55 * overlay + 0.45 * color
    )
    Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8)
    ).save(str(overlay_dir / "{}.png".format(name)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="/home/roy/data/CVC-ClinicDB",
    )
    parser.add_argument(
        "--test_list",
        default="/home/roy/data/CVC-ClinicDB/splits/test.txt",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "./model/"
            "MCSF_ResUNet_CVC_ClinicDB_bs24_256_150e/"
            "last_model.pth"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "./test_log/"
            "MCSF_ResUNet_CVC_ClinicDB_256_last"
        ),
    )
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument(
        "--inference_batch_size", type=int, default=24
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument(
        "--channels_last", type=str2bool, default=True
    )
    parser.add_argument(
        "--save_predictions", type=str2bool, default=True
    )
    return parser.parse_args()


@torch.no_grad()
def main(args):
    if args.num_classes != 2:
        raise ValueError("--num_classes must be 2.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    logger.info(
        "Arguments:\n%s",
        json.dumps(vars(args), indent=2),
    )

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    dataset = CVCClinicDBTestDataset(
        data_root=args.data_root,
        split_file=args.test_list,
        img_size=args.img_size,
    )
    if len(dataset) != 62:
        raise RuntimeError(
            "Expected 62 test images, found {}.".format(
                len(dataset)
            )
        )

    loader = DataLoader(
        dataset,
        batch_size=args.inference_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=cvc_test_collate,
        persistent_workers=(args.num_workers > 0),
    )

    model = MCSFResUNet(
        num_classes=args.num_classes,
        pretrained_path=None,
    ).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu"
    )
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(
        clean_state_dict(state), strict=True
    )
    model.eval()
    logger.info(
        "Loaded checkpoint=%s, epoch=%s",
        args.checkpoint,
        checkpoint.get("epoch", "unknown")
        if isinstance(checkpoint, dict)
        else "unknown",
    )

    rows = []
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(
            device, non_blocking=True
        )
        if args.channels_last and device.type == "cuda":
            images = images.contiguous(
                memory_format=torch.channels_last
            )

        with autocast(
            enabled=(args.amp and device.type == "cuda")
        ):
            logits = model(images)["out"]
            probabilities = torch.softmax(
                logits.float(), dim=1
            )[:, 1]

        probabilities = probabilities.cpu().numpy()

        for item_index in range(len(batch["case_name"])):
            name = batch["case_name"][item_index]
            original_size = batch["original_size"][item_index]
            target = batch["original_mask"][item_index]
            original_image = batch["original_image"][item_index]

            probability = resize_probability(
                probabilities[item_index],
                original_size,
            )
            prediction = (
                probability >= args.threshold
            ).astype(np.uint8)

            metrics = calculate_metrics(
                prediction,
                target,
                probability,
            )
            row = {"case_name": name}
            row.update(metrics)
            rows.append(row)

            if args.save_predictions:
                save_prediction_outputs(
                    output_dir,
                    name,
                    original_image,
                    target,
                    probability,
                    prediction,
                )

            logger.info(
                "[%03d/%03d] %s | Dice=%.5f | "
                "IoU=%.5f | HD95=%.3f",
                len(rows),
                len(dataset),
                name,
                metrics["dice"],
                metrics["iou"],
                metrics["hd95"],
            )

    metric_names = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "mae",
        "hd95",
    ]
    summary = {
        "checkpoint": args.checkpoint,
        "num_test_images": len(rows),
        "input_size": args.img_size,
        "threshold": args.threshold,
        "tta": False,
        "postprocessing": False,
        "metrics": {},
    }
    for metric_name in metric_names:
        values = np.asarray(
            [row[metric_name] for row in rows],
            dtype=np.float64,
        )
        summary["metrics"][metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }

    csv_path = output_dir / "per_image_metrics.csv"
    with open(str(csv_path), "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_name"] + metric_names,
        )
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Final | Dice=%.6f +/- %.6f | "
        "IoU=%.6f +/- %.6f | "
        "Accuracy=%.6f | Precision=%.6f | Recall=%.6f | "
        "MAE=%.6f | HD95=%.6f",
        summary["metrics"]["dice"]["mean"],
        summary["metrics"]["dice"]["std"],
        summary["metrics"]["iou"]["mean"],
        summary["metrics"]["iou"]["std"],
        summary["metrics"]["accuracy"]["mean"],
        summary["metrics"]["precision"]["mean"],
        summary["metrics"]["recall"]["mean"],
        summary["metrics"]["mae"]["mean"],
        summary["metrics"]["hd95"]["mean"],
    )
    

if __name__ == "__main__":
    main(parse_args())
