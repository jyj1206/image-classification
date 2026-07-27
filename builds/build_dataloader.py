import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10

from datasets.transforms import get_transforms


def build_dataloader(config, include_test=True):
    train_dataset, val_dataset, test_dataset = build_dataset(
        config, include_test=include_test
    )
    dataset_cfg = config["dataset"]
    train_cfg = dataset_cfg["train"]
    val_cfg = dataset_cfg["validation"]
    test_cfg = dataset_cfg.get("test", val_cfg)
    runtime_cfg = config["runtime"]

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    pin_memory = runtime_cfg.get("device", "cuda") == "cuda"
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    test_sampler = (
        DistributedSampler(test_dataset, shuffle=False)
        if distributed and test_dataset is not None
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=pin_memory,
        drop_last=train_cfg.get("drop_last", False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_cfg["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_cfg.get("num_workers", 0),
        pin_memory=pin_memory,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_cfg["batch_size"],
            shuffle=False,
            sampler=test_sampler,
            num_workers=test_cfg.get("num_workers", 0),
            pin_memory=pin_memory,
        )
    return train_loader, val_loader, test_loader


def build_test_dataloader(config):
    dataset_cfg = config["dataset"]
    if dataset_cfg.get("name", "").lower() != "cifar10":
        raise ValueError("This experiment supports only the CIFAR-10 dataset.")
    test_cfg = dataset_cfg.get("test", dataset_cfg["validation"])
    root = test_cfg.get(
        "root", dataset_cfg["train"].get("root", dataset_cfg.get("root", "data"))
    )
    grayscale = bool(dataset_cfg.get("grayscale", False))
    dataset = CIFAR10(
        root=root,
        train=False,
        download=test_cfg.get("download", True),
        transform=get_transforms("test", grayscale=grayscale),
    )
    pin_memory = config["runtime"].get("device", "cuda") == "cuda"
    return DataLoader(
        dataset,
        batch_size=test_cfg["batch_size"],
        shuffle=False,
        num_workers=test_cfg.get("num_workers", 0),
        pin_memory=pin_memory,
    )


def build_dataset(config, include_test=True):
    dataset_cfg = config["dataset"]
    if dataset_cfg.get("name", "").lower() != "cifar10":
        raise ValueError("This experiment supports only the CIFAR-10 dataset.")

    train_cfg = dataset_cfg["train"]
    test_cfg = dataset_cfg.get("test", dataset_cfg["validation"])
    root = train_cfg.get("root", dataset_cfg.get("root", "data"))
    seed = config["runtime"].get("seed") or 42
    grayscale = bool(dataset_cfg.get("grayscale", False))

    train_augmented = CIFAR10(
        root=root,
        train=True,
        download=train_cfg.get("download", True),
        transform=get_transforms("train", grayscale=grayscale),
    )
    train_evaluation = CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=get_transforms("validation", grayscale=grayscale),
    )
    test_dataset = None
    if include_test:
        test_dataset = CIFAR10(
            root=test_cfg.get("root", root),
            train=False,
            download=test_cfg.get("download", True),
            transform=get_transforms("test", grayscale=grayscale),
        )

    num_total = len(train_augmented)
    num_val = int(num_total * dataset_cfg.get("val_ratio", 0.1))
    indices = torch.randperm(
        num_total, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    train_indices, val_indices = indices[num_val:], indices[:num_val]
    return (
        Subset(train_augmented, train_indices),
        Subset(train_evaluation, val_indices),
        test_dataset,
    )
