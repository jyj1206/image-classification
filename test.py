"""Evaluate a trained CIFAR-10 experiment and visualize its stem filters."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from builds import build_dataloader, build_loss, build_model
from utils.utils_config import load_config, normalize_config, save_config
from utils.utils_visualization import (
    collect_class_samples,
    save_stem_feature_maps,
    save_stem_visualization,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR ResNet-20 and visualize its stem filters."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment YAML. Omit when --checkpoint is a result directory.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint file or result directory containing checkpoints/best.pth.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to the checkpoint/result directory.",
    )
    return parser.parse_args()


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoints" / "best.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def create_test_run_dir(config):
    experiment_name = config["experiment"]["name"]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(experiment_name)).strip("._-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path("runs")
    run_dir = run_root / f"run_{safe_name}_cifar10_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = run_root / f"run_{safe_name}_cifar10_{timestamp}_{suffix:03d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    with torch.inference_mode():
        progress = tqdm(loader, desc="Test", dynamic_ncols=True)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += labels.size(0)
            progress.set_postfix(
                loss=f"{loss_sum / sample_count:.4f}",
                accuracy=f"{100.0 * correct / sample_count:.2f}%",
            )

    return {
        "loss": loss_sum / sample_count,
        "accuracy": 100.0 * correct / sample_count,
        "correct": correct,
        "total": sample_count,
    }


def main():
    args = parse_args()
    checkpoint_argument = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else None
    if config_path is None and checkpoint_argument.is_dir():
        config_path = checkpoint_argument / "config.yaml"
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(
            "Config was not found. Pass --config or provide a result directory "
            "containing config.yaml as --checkpoint."
        )

    config = normalize_config(load_config(config_path))
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    output_dir = Path(args.output) if args.output else create_test_run_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = config["runtime"].get("device", "cuda")
    device = torch.device(
        "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    criterion = build_loss(config)
    _, _, test_loader = build_dataloader(config)
    visualization_images, visualization_labels = collect_class_samples(test_loader)
    visualization_images = visualization_images.to(device, non_blocking=True)
    metrics = evaluate(model, test_loader, criterion, device)

    visualization_dir = output_dir / "visualizations"
    visualization_path = (
        visualization_dir / "kernels" / "test" / "kernel_overview.png"
    )
    feature_map_path = (
        visualization_dir / "feature_maps" / "test" / "feature_map_overview.png"
    )
    metrics_path = output_dir / "test_metrics.json"
    filter_type = config["model"]["args"]["stem_filter"]
    save_stem_visualization(
        model.stem_conv.weight,
        visualization_path,
        title="Stem filters at test time",
        filter_type=filter_type,
    )
    save_stem_feature_maps(
        model,
        visualization_images,
        feature_map_path,
        title="Test feature maps after depthwise 3x3",
        filter_type=filter_type,
        sample_labels=visualization_labels,
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["config"] = str(config_path)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    save_config(config, output_dir / "config.yaml")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Test loss: {metrics['loss']:.4f}")
    print(
        f"Test accuracy: {metrics['accuracy']:.2f}% "
        f"({metrics['correct']}/{metrics['total']})"
    )
    print(f"Filter visualization: {visualization_path}")
    print(f"Feature-map visualization: {feature_map_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
