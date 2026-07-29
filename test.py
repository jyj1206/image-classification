"""Evaluate one trained CIFAR-10 experiment and save metrics/visualizations."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from builds import build_loss, build_model, build_test_dataloader
from utils.utils_config import load_config, normalize_config, save_config
from utils.utils_visualization import (
    collect_class_samples,
    save_stem_feature_maps,
    save_stem_visualization,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one CIFAR-10 checkpoint.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint file or result directory containing checkpoints/best.pth.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None, choices=("cuda", "cpu"))
    return parser.parse_args()


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoints" / "best.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def resolve_config_path(config_path, checkpoint_argument):
    if config_path:
        path = Path(config_path)
    else:
        argument = Path(checkpoint_argument)
        result_dir = argument if argument.is_dir() else argument.parent.parent
        path = result_dir / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")
    return path


def create_test_run_dir(config, run_root="runs"):
    experiment_name = config["experiment"]["name"]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(experiment_name)).strip("._-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(run_root)
    run_dir = run_root / f"run_{safe_name}_cifar10_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = run_root / f"run_{safe_name}_cifar10_{timestamp}_{suffix:03d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def evaluate(model, loader, criterion, device, show_progress=True):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)

    with torch.inference_mode():
        progress = tqdm(
            loader, desc="Test", dynamic_ncols=True, disable=not show_progress
        )
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            loss_sum += loss.item() * labels.size(0)
            sample_count += labels.size(0)

            flat_indices = (labels * 10 + predictions).detach().cpu()
            confusion += torch.bincount(flat_indices, minlength=100).reshape(10, 10)
            correct = int(confusion.diag().sum())
            progress.set_postfix(
                loss=f"{loss_sum / sample_count:.4f}",
                accuracy=f"{100.0 * correct / sample_count:.2f}%",
            )

    true_positive = confusion.diag().float()
    predicted_count = confusion.sum(dim=0).float()
    actual_count = confusion.sum(dim=1).float()
    precision_per_class = torch.where(
        predicted_count > 0, true_positive / predicted_count, 0.0
    )
    recall_per_class = torch.where(
        actual_count > 0, true_positive / actual_count, 0.0
    )
    f1_per_class = torch.where(
        precision_per_class + recall_per_class > 0,
        2 * precision_per_class * recall_per_class
        / (precision_per_class + recall_per_class),
        0.0,
    )
    correct = int(true_positive.sum())
    return {
        "loss": loss_sum / sample_count,
        "accuracy": 100.0 * correct / sample_count,
        "precision": 100.0 * precision_per_class.mean().item(),
        "recall": 100.0 * recall_per_class.mean().item(),
        "f1_score": 100.0 * f1_per_class.mean().item(),
        "averaging": "macro",
        "correct": correct,
        "total": sample_count,
        "confusion_matrix": confusion.tolist(),
    }


def run_test(
    config_path,
    checkpoint_path,
    output_dir=None,
    device_name=None,
    show_progress=True,
    verbose=True,
    runs_root="runs",
):
    config_path = Path(config_path)
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    config = normalize_config(load_config(config_path))
    output_dir = (
        Path(output_dir) if output_dir else create_test_run_dir(config, runs_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = device_name or config["runtime"].get("device", "cuda")
    device = torch.device(
        "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    criterion = build_loss(config)
    test_loader = build_test_dataloader(config)
    visualization_images, visualization_labels = collect_class_samples(test_loader)
    visualization_images = visualization_images.to(device, non_blocking=True)
    metrics = evaluate(model, test_loader, criterion, device, show_progress)

    visualization_dir = output_dir / "visualizations"
    kernel_path = visualization_dir / "kernels" / "test" / "kernel_overview.png"
    feature_path = (
        visualization_dir / "feature_maps" / "test" / "feature_map_overview.png"
    )
    filter_type = config["model"]["args"]["stem_filter"]
    save_stem_visualization(
        model.stem_conv.weight,
        kernel_path,
        title="Stem filters at test time",
        filter_type=filter_type,
        input_channels=config["model"]["args"]["input_channels"],
    )
    save_stem_feature_maps(
        model,
        visualization_images,
        feature_path,
        title="Test feature maps after depthwise 3x3",
        filter_type=filter_type,
        sample_labels=visualization_labels,
    )

    metrics.update({
        "experiment": config["experiment"]["name"],
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
    })
    metrics_path = output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    save_config(config, output_dir / "config.yaml")

    if verbose:
        print(
            f"Experiment: {metrics['experiment']} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.2f}% | "
            f"Precision (macro): {metrics['precision']:.2f}% | "
            f"Recall (macro): {metrics['recall']:.2f}% | "
            f"F1 (macro): {metrics['f1_score']:.2f}%"
        )
        print(f"Run directory: {output_dir}")
    return metrics, output_dir


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config, args.checkpoint)
    run_test(
        config_path=config_path,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
