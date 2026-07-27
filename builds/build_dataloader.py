import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from datasets import PairedImageDataset, get_transforms


def build_dataloader(config):
    train_dataset, val_dataset, test_dataset = build_dataset(config)

    train_config = config["dataset"]["train"]
    test_config = config["dataset"]["validation"]
    runtime_config = config["runtime"]

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    pin_memory = runtime_config.get("device", "cuda") == "cuda"

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_config.get("num_workers", 0),
        pin_memory=pin_memory,
        drop_last=train_config.get("drop_last", False),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=test_config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=test_config.get("num_workers", 0),
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_config["batch_size"],
        shuffle=False,
        sampler=test_sampler,
        num_workers=test_config.get("num_workers", 0),
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def build_dataset(config):
    train_config = config["dataset"]["train"]
    test_config = config["dataset"]["validation"]

    val_ratio = config["dataset"].get("val_ratio", 0.1)
    seed = config["runtime"].get("seed", 42)
    seed = 42 if seed is None else seed

    train_source = PairedImageDataset(
        root_dir=train_config["root"],
        split=train_config["split"],
        patch_size=train_config.get("patch_size"),
        transform=get_transforms(split="train"),
    )

    val_source = PairedImageDataset(
        root_dir=train_config["root"],
        split=train_config["split"],
        patch_size=None,
        transform=get_transforms(split="validation"),
    )

    test_dataset = PairedImageDataset(
        root_dir=test_config["root"],
        split=test_config["split"],
        patch_size=None,
        transform=get_transforms(split="validation"),
    )

    num_total = len(train_source)
    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_total, generator=generator).tolist()

    train_dataset = Subset(train_source, indices[:num_train])
    val_dataset = Subset(val_source, indices[num_train:])

    return train_dataset, val_dataset, test_dataset
