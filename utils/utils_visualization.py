from pathlib import Path

import torch

def save_stem_visualization(weight, path, title="Stem filters", filter_type=None):
    """Save three ordinary or six Sobel depthwise kernels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weight = weight.detach().float().cpu()
    if weight.ndim != 4 or weight.shape[0] not in (3, 6, 12) or tuple(weight.shape[1:]) != (1, 3, 3):
        raise ValueError(
            "Expected 3, 6, or 12 depthwise 3x3 kernels, "
            f"got {tuple(weight.shape)}"
        )
    spatial_filters = weight[:, 0]
    limit = max(float(spatial_filters.abs().max()), 1e-8)
    kernel_count = spatial_filters.shape[0]
    if kernel_count == 12:
        figure, axes = plt.subplots(3, 4, figsize=(9, 7))
    else:
        figure, axes = plt.subplots(1, kernel_count, figsize=(2.5 * kernel_count, 3))
    axes = axes.reshape(-1)
    image = None
    if kernel_count == 3:
        channel_names = ("Red", "Green", "Blue")
    elif filter_type == "sobel":
        channel_names = ("Red-X", "Red-Y", "Green-X", "Green-Y", "Blue-X", "Blue-Y")
    elif filter_type == "mixed":
        channel_names = tuple(
            f"{channel}-{kernel}"
            for channel in ("Red", "Green", "Blue")
            for kernel in ("Sobel-X", "Sobel-Y", "Laplacian", "Gaussian")
        )
    else:
        multiplier = kernel_count // 3
        channel_names = tuple(
            f"{channel}-{index + 1}"
            for channel in ("Red", "Green", "Blue")
            for index in range(multiplier)
        )
    for channel_index, axis in enumerate(axes):
        image = axis.imshow(
            spatial_filters[channel_index],
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(channel_names[channel_index])
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.colorbar(image, ax=axes, fraction=0.015, pad=0.02)
    figure.subplots_adjust(top=0.78, right=0.9, wspace=0.2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_stem_feature_maps(
    model,
    images,
    path,
    title="Feature maps after depthwise 3x3",
    filter_type=None,
    max_samples=4,
):
    """Visualize inputs and outputs immediately after the first convolution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = images[:max_samples]
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        feature_maps = model.stem_conv(images).detach().float().cpu()
    model.train(was_training)
    inputs = images.detach().float().cpu()

    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
    inputs = (inputs * std + mean).clamp(0, 1)

    kernel_count = feature_maps.shape[1]
    if kernel_count == 3:
        map_names = ("Red", "Green", "Blue")
    elif filter_type == "sobel":
        map_names = ("Red-X", "Red-Y", "Green-X", "Green-Y", "Blue-X", "Blue-Y")
    elif filter_type == "mixed":
        map_names = tuple(
            f"{channel}-{kernel}"
            for channel in ("R", "G", "B")
            for kernel in ("Sx", "Sy", "Lap", "Gau")
        )
    else:
        multiplier = kernel_count // 3
        map_names = tuple(
            f"{channel}-{index + 1}"
            for channel in ("R", "G", "B")
            for index in range(multiplier)
        )

    limit = max(float(feature_maps.abs().max()), 1e-8)
    rows = inputs.shape[0]
    figure, axes = plt.subplots(
        rows, kernel_count + 1,
        figsize=(2.2 * (kernel_count + 1), 2.2 * rows),
        squeeze=False,
    )
    map_image = None
    for row in range(rows):
        axes[row, 0].imshow(inputs[row].permute(1, 2, 0))
        axes[row, 0].set_ylabel(f"sample {row}", fontsize=8)
        if row == 0:
            axes[row, 0].set_title("Input")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for channel in range(kernel_count):
            axis = axes[row, channel + 1]
            map_image = axis.imshow(
                feature_maps[row, channel],
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            if row == 0:
                axis.set_title(map_names[channel], fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(title)
    figure.colorbar(map_image, ax=axes[:, 1:], fraction=0.012, pad=0.02)
    figure.subplots_adjust(top=0.9, right=0.92, hspace=0.12, wspace=0.08)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [item["train_accuracy"] for item in history], label="train")
    axes[1].plot(epochs, [item["val_accuracy"] for item in history], label="validation")
    axes[1].set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
