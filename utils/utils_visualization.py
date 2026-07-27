from pathlib import Path
import re

import torch


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def collect_class_samples(loader, num_classes=10):
    """Collect the first deterministic sample of each class from a loader."""
    samples = {}
    for images, labels in loader:
        for image, label in zip(images, labels):
            class_index = int(label)
            if class_index not in samples:
                samples[class_index] = image
            if len(samples) == num_classes:
                ordered_labels = list(range(num_classes))
                return (
                    torch.stack([samples[index] for index in ordered_labels]),
                    ordered_labels,
                )
    missing = sorted(set(range(num_classes)) - set(samples))
    raise RuntimeError(f"Could not collect visualization samples for classes: {missing}")


def _filter_layout(kernel_count, filter_type):
    channels = ("R", "G", "B")
    if kernel_count in (1, 2, 4):
        if filter_type == "sobel":
            operators = ("Sobel_X", "Sobel_Y")
        elif filter_type == "mixed":
            operators = ("Sobel_X", "Sobel_Y", "Laplacian", "Gaussian")
        elif kernel_count == 1:
            operators = (
                filter_type.capitalize() if filter_type != "learnable" else "Random",
            )
        else:
            operators = tuple(f"Random_{index + 1}" for index in range(kernel_count))
        raw_names = tuple(f"Gray_{operator}" for operator in operators)
        groups = tuple((operator, (index,)) for index, operator in enumerate(operators))
        return raw_names, groups

    if kernel_count == 3:
        raw_names = channels
        group_name = filter_type.capitalize() if filter_type != "learnable" else "Random"
        return raw_names, ((group_name, (0, 1, 2)),)

    if filter_type == "sobel":
        operators = ("Sobel_X", "Sobel_Y")
    elif filter_type == "mixed":
        operators = ("Sobel_X", "Sobel_Y", "Laplacian", "Gaussian")
    else:
        operators = tuple(f"Random_{index + 1}" for index in range(kernel_count // 3))

    multiplier = len(operators)
    raw_names = tuple(
        f"{channel}_{operator}"
        for channel in channels
        for operator in operators
    )
    groups = tuple(
        (
            operator,
            tuple(channel * multiplier + operator_index for channel in range(3)),
        )
        for operator_index, operator in enumerate(operators)
    )
    return raw_names, groups


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def _asset_stem(path):
    return path.stem[:-9] if path.stem.endswith("_overview") else path.stem


def _save_heatmap(values, path, title, limit, dpi=180):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(3, 3))
    image = axis.imshow(
        values,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    if values.numel() <= 9:
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{float(values[row, column]):.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight", transparent=False)
    plt.close(figure)


def save_stem_visualization(weight, path, title="Stem filters", filter_type=None):
    """Save one overview plus one PPT-ready image per depthwise kernel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weight = weight.detach().float().cpu()
    if (
        weight.ndim != 4
        or weight.shape[0] not in (1, 2, 3, 4, 6, 12)
        or tuple(weight.shape[1:]) != (1, 3, 3)
    ):
        raise ValueError(f"Expected 1, 2, 3, 4, 6, or 12 depthwise kernels, got {tuple(weight.shape)}")

    filters = weight[:, 0]
    kernel_count = filters.shape[0]
    raw_names, _ = _filter_layout(kernel_count, filter_type)
    limit = max(float(filters.abs().max()), 1e-8)
    rows, columns = ((3, 4) if kernel_count == 12 else (1, kernel_count))
    figure, axes = plt.subplots(rows, columns, figsize=(2.5 * columns, 2.5 * rows), squeeze=False)
    image = None
    for index, axis in enumerate(axes.flat):
        image = axis.imshow(filters[index], cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(raw_names[index].replace("_", " "), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.colorbar(image, ax=axes, fraction=0.015, pad=0.02)
    figure.subplots_adjust(top=0.88, right=0.9, hspace=0.25, wspace=0.15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    asset_stem = _asset_stem(path)
    for index, name in enumerate(raw_names):
        individual_path = path.with_name(f"{asset_stem}_{_safe_name(name)}{path.suffix}")
        _save_heatmap(filters[index], individual_path, name.replace("_", " "), limit)


def save_stem_feature_maps(
    model,
    images,
    path,
    title="Feature maps after depthwise 3x3",
    filter_type=None,
    sample_labels=None,
    max_samples=10,
):
    """Save an RGB-composite overview and PPT-ready input/raw/RGB images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = images[:max_samples]
    if sample_labels is None:
        sample_labels = list(range(len(images)))
    sample_labels = list(sample_labels)[: len(images)]
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        feature_maps = model.stem_conv(images).detach().float().cpu()
    model.train(was_training)

    inputs = images.detach().float().cpu()
    if inputs.shape[1] == 1:
        mean = torch.tensor((0.4809,)).view(1, 1, 1, 1)
        std = torch.tensor((0.2392,)).view(1, 1, 1, 1)
    else:
        mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
        std = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
    inputs = (inputs * std + mean).clamp(0, 1)

    raw_names, groups = _filter_layout(feature_maps.shape[1], filter_type)
    limit = max(float(feature_maps.abs().max()), 1e-8)
    asset_stem = _asset_stem(path)
    rows, columns = inputs.shape[0], len(groups) + 1
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.5 * columns, 2.5 * rows), squeeze=False
    )

    for sample_index in range(rows):
        class_index = int(sample_labels[sample_index])
        class_name = (
            CIFAR10_CLASSES[class_index]
            if 0 <= class_index < len(CIFAR10_CLASSES)
            else f"class_{class_index}"
        )
        sample_name = f"class{class_index:02d}_{class_name}"
        if inputs.shape[1] == 1:
            input_image = inputs[sample_index, 0].numpy()
            axes[sample_index, 0].imshow(input_image, cmap="gray", vmin=0, vmax=1)
        else:
            input_image = inputs[sample_index].permute(1, 2, 0).numpy()
            axes[sample_index, 0].imshow(input_image)
        axes[sample_index, 0].set_ylabel(
            f"{class_index}: {class_name}", fontsize=8
        )
        if sample_index == 0:
            axes[sample_index, 0].set_title("Input")
        axes[sample_index, 0].set_xticks([])
        axes[sample_index, 0].set_yticks([])

        input_path = path.with_name(f"{asset_stem}_{sample_name}_input.png")
        plt.imsave(
            input_path,
            input_image,
            cmap="gray" if inputs.shape[1] == 1 else None,
            vmin=0 if inputs.shape[1] == 1 else None,
            vmax=1 if inputs.shape[1] == 1 else None,
        )

        for raw_index, raw_name in enumerate(raw_names):
            raw_path = path.with_name(
                f"{asset_stem}_{sample_name}_{_safe_name(raw_name)}.png"
            )
            _save_heatmap(
                feature_maps[sample_index, raw_index],
                raw_path,
                raw_name.replace("_", " "),
                limit,
                dpi=160,
            )

        for group_index, (group_name, indices) in enumerate(groups):
            axis = axes[sample_index, group_index + 1]
            if len(indices) == 3:
                response = feature_maps[sample_index, list(indices)].permute(1, 2, 0)
                display = (response / (2.0 * limit) + 0.5).clamp(0, 1).numpy()
                axis.imshow(display)
                summary_kind = "RGB"
            else:
                display = feature_maps[sample_index, indices[0]].numpy()
                axis.imshow(display, cmap="coolwarm", vmin=-limit, vmax=limit)
                summary_kind = "Gray"
            if sample_index == 0:
                axis.set_title(
                    f"{group_name.replace('_', ' ')} {summary_kind}", fontsize=9
                )
            axis.set_xticks([])
            axis.set_yticks([])
            if summary_kind == "RGB":
                rgb_path = path.with_name(
                    f"{asset_stem}_{sample_name}_RGB_{_safe_name(group_name)}.png"
                )
                plt.imsave(rgb_path, display)

    figure.suptitle(title)
    figure.subplots_adjust(top=0.9, hspace=0.12, wspace=0.08)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    best = max(history, key=lambda item: item["val_accuracy"])
    best_epoch = best["epoch"]
    best_accuracy = best["val_accuracy"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="validation")
    axes[0].axvline(best_epoch, color="red", linestyle="--", alpha=0.6, label="best epoch")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [item["train_accuracy"] for item in history], label="train")
    axes[1].plot(epochs, [item["val_accuracy"] for item in history], label="validation")
    axes[1].scatter(
        [best_epoch], [best_accuracy], color="red", marker="*", s=130, zorder=5,
        label=f"best: {best_accuracy:.2f}% @ {best_epoch}",
    )
    axes[1].axvline(best_epoch, color="red", linestyle="--", alpha=0.6)
    axes[1].set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")
    axes[1].legend()
    axes[2].plot(epochs, [item["lr"] for item in history], color="tab:green")
    axes[2].axvline(best_epoch, color="red", linestyle="--", alpha=0.6)
    axes[2].set(xlabel="epoch", ylabel="learning rate", title="Learning Rate")
    axes[2].set_yscale("log")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
