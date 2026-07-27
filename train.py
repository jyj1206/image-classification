import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from builds import (
    build_dataloader,
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
)
from utils import (
    load_train_config,
    prepare_checkpoint_dir,
    prepare_result_dir,
    save_result_config,
)
from utils.utils_dist import (
    cleanup_distributed,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    unwrap_model,
)
from utils.utils_visualization import (
    save_stem_feature_maps,
    save_stem_visualization,
    save_training_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 stem-filter experiment")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", default=None, help="checkpoint or result directory")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, device, optimizer=None, description=None):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(3, dtype=torch.float64, device=device)

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
        disable=not is_main_process(),
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        totals[0] += loss.detach() * labels.size(0)
        totals[1] += (logits.argmax(1) == labels).sum()
        totals[2] += labels.size(0)
        progress.set_postfix(
            loss=f"{(totals[0] / totals[2]).item():.4f}",
            acc=f"{(100.0 * totals[1] / totals[2]).item():.2f}%",
        )

    if is_distributed():
        torch.distributed.all_reduce(totals)
    return {
        "loss": (totals[0] / totals[2]).item(),
        "accuracy": (100.0 * totals[1] / totals[2]).item(),
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_accuracy, history):
    torch.save({
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_accuracy": best_accuracy,
        "history": history,
    }, path)


def resolve_resume_path(resume):
    if not resume:
        return None
    path = Path(resume)
    return path / "checkpoints" / "latest.pth" if path.is_dir() else path


def write_history(history, result_dir):
    with (result_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (result_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_training_curves(history, result_dir / "training_curves.png")


def main():
    args = parse_args()
    config, result_dir = load_train_config(args.config, args.resume)
    dist_info = init_distributed(config)
    device = dist_info["device"]
    seed_everything(int(config["runtime"].get("seed") or 42))

    try:
        result_dir = prepare_result_dir(config, result_dir)
        checkpoint_dir = prepare_checkpoint_dir(result_dir, config)
        save_result_config(config, result_dir)

        train_loader, val_loader, _ = build_dataloader(config, include_test=False)
        model = build_model(config).to(device)
        criterion = build_loss(config)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer)

        start_epoch, best_accuracy, history = 1, -1.0, []
        resume_path = resolve_resume_path(args.resume)
        if resume_path:
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            best_accuracy = checkpoint.get("best_accuracy", -1.0)
            history = checkpoint.get("history", [])

        if is_main_process():
            visualization_images = next(iter(val_loader))[0][:4].to(
                device, non_blocking=True
            )
            filter_type = config["model"]["args"]["stem_filter"]
            save_stem_visualization(
                model.stem_conv.weight,
                result_dir / "stem_filters_initial.png",
                "Initial stem filters",
                filter_type=filter_type,
            )
            save_stem_feature_maps(
                model,
                visualization_images,
                result_dir / "stem_feature_maps_initial.png",
                "Initial feature maps after depthwise 3x3",
                filter_type=filter_type,
            )
            trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            total = sum(parameter.numel() for parameter in model.parameters())
            print(f"Result directory: {result_dir}")
            print(f"Device: {device} | world size: {get_world_size()}")
            print(f"Parameters: {trainable:,} trainable / {total:,} total")

        if is_distributed():
            model = DistributedDataParallel(
                model,
                device_ids=[dist_info["local_rank"]] if device.type == "cuda" else None,
            )

        epochs = int(config["trainer"]["epochs"])
        for epoch in range(start_epoch, epochs + 1):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model, train_loader, criterion, device, optimizer,
                description=f"Epoch {epoch:03d}/{epochs} train",
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, device,
                description=f"Epoch {epoch:03d}/{epochs} val",
            )
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(record)
            scheduler.step()
            if is_main_process():
                print(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"train {record['train_loss']:.4f}, {record['train_accuracy']:.2f}% | "
                    f"val {record['val_loss']:.4f}, {record['val_accuracy']:.2f}%"
                )
                is_best = record["val_accuracy"] > best_accuracy
                best_accuracy = max(best_accuracy, record["val_accuracy"])
                save_checkpoint(
                    checkpoint_dir / "latest.pth", model, optimizer, scheduler,
                    epoch, best_accuracy, history,
                )
                if is_best:
                    save_checkpoint(
                        checkpoint_dir / "best.pth", model, optimizer, scheduler,
                        epoch, best_accuracy, history,
                    )
                write_history(history, result_dir)

        if is_distributed():
            torch.distributed.barrier()
        best = torch.load(checkpoint_dir / "best.pth", map_location=device)
        unwrap_model(model).load_state_dict(best["model"])
        if is_main_process():
            save_stem_visualization(
                unwrap_model(model).stem_conv.weight,
                result_dir / "stem_filters_final.png",
                "Final stem filters (best checkpoint)",
                filter_type=filter_type,
            )
            save_stem_feature_maps(
                unwrap_model(model),
                visualization_images,
                result_dir / "stem_feature_maps_final.png",
                "Final feature maps after depthwise 3x3 (best checkpoint)",
                filter_type=filter_type,
            )
            print(f"Best validation accuracy: {best_accuracy:.2f}%")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
