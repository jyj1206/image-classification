from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_GRAYSCALE_MEAN = (0.4809,)
CIFAR10_GRAYSCALE_STD = (0.2392,)


def get_transforms(split="train", grayscale=False):
    color_transform = [transforms.Grayscale(num_output_channels=1)] if grayscale else []
    normalize = transforms.Normalize(
        mean=CIFAR10_GRAYSCALE_MEAN if grayscale else CIFAR10_MEAN,
        std=CIFAR10_GRAYSCALE_STD if grayscale else CIFAR10_STD,
    )
    if split == "train":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            *color_transform,
            transforms.ToTensor(),
            normalize,
        ])
    if split in ("val", "validation", "test"):
        return transforms.Compose([*color_transform, transforms.ToTensor(), normalize])
    raise ValueError(f"Unsupported split: {split}")
