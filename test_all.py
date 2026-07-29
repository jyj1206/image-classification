"""Evaluate every best checkpoint under results/ and write an aggregate CSV."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from test import run_test
from utils.utils_config import load_config, normalize_config


CSV_FIELDS = (
    "experiment",
    "experiment_description",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "loss",
    "total_parameters",
    "trainable_parameters",
    "macs",
    "macs_m",
    "flops",
    "flops_m",
    "complexity_input_shape",
    "flops_convention",
    "averaging",
    "result_directory",
    "test_run_directory",
    "checkpoint",
    "status",
    "error",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate <results-root>/<result-directory>/checkpoints/best.pth files."
        )
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def describe_experiment(config):
    model_args = config["model"]["args"]
    color = "Grayscale" if config["dataset"].get("grayscale", False) else "RGB"
    normalization = (
        "BatchNorm" if model_args.get("use_batchnorm", True) else "No BatchNorm"
    )
    kernel_count = model_args["spatial_kernels"]
    filter_type = model_args["stem_filter"]
    filter_name = "Random" if filter_type == "learnable" else filter_type.capitalize()
    trainability = "Learnable" if model_args["stem_trainable"] else "Fixed"
    return (
        f"{color} | {normalization} | {kernel_count} kernels | "
        f"{filter_name} | {trainability}"
    )


def write_summary(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    checkpoints = sorted(results_root.glob("*/checkpoints/best.pth"))
    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints matching "
            f"{results_root}/*/checkpoints/best.pth were found."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = Path(args.runs_root)
    csv_path = (
        Path(args.output_csv)
        if args.output_csv
        else runs_root / f"all_test_results_{timestamp}.csv"
    )
    rows = []
    print(f"Found {len(checkpoints)} best checkpoint(s) under {results_root}")

    for index, checkpoint_path in enumerate(checkpoints, start=1):
        result_dir = checkpoint_path.parent.parent
        config_path = result_dir / "config.yaml"
        print(f"\n[{index}/{len(checkpoints)}] Testing: {result_dir}")
        try:
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config: {config_path}")
            config = normalize_config(load_config(config_path))
            metrics, run_dir = run_test(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                output_dir=None,
                device_name=args.device,
                show_progress=True,
                verbose=True,
                runs_root=runs_root,
            )
            row = {
                "experiment": config["experiment"]["name"],
                "experiment_description": describe_experiment(config),
                "accuracy": f"{metrics['accuracy']:.6f}",
                "precision": f"{metrics['precision']:.6f}",
                "recall": f"{metrics['recall']:.6f}",
                "f1_score": f"{metrics['f1_score']:.6f}",
                "loss": f"{metrics['loss']:.8f}",
                "total_parameters": metrics["total_parameters"],
                "trainable_parameters": metrics["trainable_parameters"],
                "macs": metrics["macs"],
                "macs_m": f"{metrics['macs_m']:.6f}",
                "flops": metrics["flops"],
                "flops_m": f"{metrics['flops_m']:.6f}",
                "complexity_input_shape": metrics["complexity_input_shape"],
                "flops_convention": metrics["flops_convention"],
                "averaging": metrics["averaging"],
                "result_directory": str(result_dir),
                "test_run_directory": str(run_dir),
                "checkpoint": str(checkpoint_path),
                "status": "success",
                "error": "",
            }
        except Exception as error:
            row = {
                "experiment": result_dir.name,
                "experiment_description": "",
                "accuracy": "",
                "precision": "",
                "recall": "",
                "f1_score": "",
                "loss": "",
                "total_parameters": "",
                "trainable_parameters": "",
                "macs": "",
                "macs_m": "",
                "flops": "",
                "flops_m": "",
                "complexity_input_shape": "",
                "flops_convention": "",
                "averaging": "macro",
                "result_directory": str(result_dir),
                "test_run_directory": "",
                "checkpoint": str(checkpoint_path),
                "status": "failed",
                "error": str(error),
            }
            print(f"FAILED: {error}")
            if args.fail_fast:
                rows.append(row)
                write_summary(rows, csv_path)
                raise
        rows.append(row)
        write_summary(rows, csv_path)

    success_count = sum(row["status"] == "success" for row in rows)
    print(
        f"\nCompleted: {success_count}/{len(rows)} succeeded | "
        f"Summary CSV: {csv_path}"
    )


if __name__ == "__main__":
    main()
