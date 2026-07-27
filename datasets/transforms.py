from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(split="train"):
    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    if split == "train":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    if split in ("val", "validation", "test"):
        return transforms.Compose([transforms.ToTensor(), normalize])
    raise ValueError(f"Unsupported split: {split}")
