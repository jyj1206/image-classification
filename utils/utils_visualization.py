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


def _filter_layout(
    kernel_count, filter_type, kernel_view=False, input_channels=None
):
    channels = ("R", "G", "B")
    if filter_type == "sobel_magnitude":
        input_channels = input_channels or (1 if kernel_count in (1, 2) else 3)
        if kernel_view:
            operators = ("Sobel_X", "Sobel_Y")
            raw_names = tuple(
                f"{channel}_{operator}"
                for channel in channels[:input_channels]
                for operator in operators
            )
            return raw_names, ()
        raw_names = tuple(
            f"{channel}_Sobel_Magnitude" for channel in channels[:input_channels]
        )
        groups = (
            (("Sobel_Magnitude", tuple(range(3))),)
            if input_channels == 3
            else (("Sobel_Magnitude", (0,)),)
        )
        return raw_names, groups

    if filter_type == "mixed_magnitude":
        input_channels = input_channels or (1 if kernel_count in (3, 4) else 3)
        if kernel_view:
            operators = ("Identity", "Laplacian", "Sobel_X", "Sobel_Y")
        else:
            operators = ("Identity", "Laplacian", "Sobel_Magnitude")
        raw_names = tuple(
            f"{channel}_{operator}"
            for channel in channels[:input_channels]
            for operator in operators
        )
        if kernel_view:
            return raw_names, ()
        multiplier = len(operators)
        groups = tuple(
            (
                operator,
                tuple(
                    channel * multiplier + operator_index
                    for channel in range(input_channels)
                ),
            )
            for operator_index, operator in enumerate(operators)
        )
        return raw_names, groups

    if input_channels == 1 or (
        input_channels is None and kernel_count in (1, 2, 4)
    ):
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


def _save_nearest_variants(path):
    """Save 2x/4x nearest-neighbor copies in dedicated sibling folders."""
    from PIL import Image

    path = Path(path)
    with Image.open(path) as image:
        for scale in (2, 4):
            scaled_dir = path.parent / f"nearest_{scale}x"
            scaled_dir.mkdir(parents=True, exist_ok=True)
            scaled = image.resize(
                (image.width * scale, image.height * scale),
                resample=Image.Resampling.NEAREST,
            )
            scaled.save(scaled_dir / path.name)


def _response_image(values, limit):
    """Map a signed response to display gray while keeping zero at mid-gray."""
    return (values / (2.0 * limit) + 0.5).clamp(0, 1).numpy()


def _as_rgb(image):
    import numpy as np

    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    return image


def _save_input_output_pair(input_image, output_image, path):
    """Save a title-free input/output pair suited for direct PPT insertion."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left = _as_rgb(input_image)
    right = _as_rgb(output_image)
    gap = np.ones((left.shape[0], 2, 3), dtype=np.float32)
    paired = np.concatenate((left, gap, right), axis=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, paired)
    _save_nearest_variants(path)


def _save_heatmap(values, path, title, limit, dpi=180, colorbar=True):
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
    if colorbar:
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight", transparent=False)
    plt.close(figure)


def save_stem_visualization(
    weight, path, title="Stem filters", filter_type=None, input_channels=None
):
    """Save one overview plus one PPT-ready image per depthwise kernel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weight = weight.detach().float().cpu()
    if (
        weight.ndim != 4
        or weight.shape[0] not in (1, 2, 3, 4, 6, 9, 12)
        or tuple(weight.shape[1:]) != (1, 3, 3)
    ):
        raise ValueError(
            "Expected 1, 2, 3, 4, 6, 9, or 12 depthwise kernels, "
            f"got {tuple(weight.shape)}"
        )

    filters = weight[:, 0]
    kernel_count = filters.shape[0]
    raw_names, _ = _filter_layout(
        kernel_count,
        filter_type,
        kernel_view=True,
        input_channels=input_channels,
    )
    limit = max(float(filters.abs().max()), 1e-8)
    if kernel_count == 12:
        rows, columns = 3, 4
    elif kernel_count == 9:
        rows, columns = 3, 3
    else:
        rows, columns = 1, kernel_count
    figure, axes = plt.subplots(rows, columns, figsize=(2.5 * columns, 2.5 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.imshow(filters[index], cmap="coolwarm", vmin=-limit, vmax=limit)
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{float(filters[index, row, column]):.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        axis.set_title(raw_names[index].replace("_", " "), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    if title:
        figure.suptitle(title)
    figure.subplots_adjust(top=0.88, hspace=0.25, wspace=0.15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    asset_stem = _asset_stem(path)
    for index, name in enumerate(raw_names):
        individual_path = path.with_name(f"{asset_stem}_{_safe_name(name)}{path.suffix}")
        _save_heatmap(
            filters[index],
            individual_path,
            name.replace("_", " "),
            limit,
            colorbar=False,
        )


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

    raw_names, groups = _filter_layout(
        feature_maps.shape[1],
        filter_type,
        input_channels=inputs.shape[1],
    )
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
        class_dir = path.parent / sample_name
        class_dir.mkdir(parents=True, exist_ok=True)
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

        input_path = class_dir / f"{asset_stem}_input.png"
        plt.imsave(
            input_path,
            input_image,
            cmap="gray" if inputs.shape[1] == 1 else None,
            vmin=0 if inputs.shape[1] == 1 else None,
            vmax=1 if inputs.shape[1] == 1 else None,
        )
        _save_nearest_variants(input_path)
        if inputs.shape[1] == 3:
            for channel_index, channel_name in enumerate(("R", "G", "B")):
                channel_path = class_dir / (
                    f"{asset_stem}_input_{channel_name}.png"
                )
                plt.imsave(
                    channel_path,
                    inputs[sample_index, channel_index].numpy(),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                _save_nearest_variants(channel_path)

        for raw_index, raw_name in enumerate(raw_names):
            raw_path = class_dir / (
                f"{asset_stem}_{_safe_name(raw_name)}.png"
            )
            raw_display = _response_image(
                feature_maps[sample_index, raw_index], limit
            )
            plt.imsave(raw_path, raw_display, cmap="gray", vmin=0, vmax=1)
            _save_nearest_variants(raw_path)

            if inputs.shape[1] == 3:
                source_channel = raw_index // max(
                    feature_maps.shape[1] // inputs.shape[1], 1
                )
                pair_input = inputs[sample_index, source_channel].numpy()
            else:
                pair_input = inputs[sample_index, 0].numpy()
            pair_path = class_dir / "pairs" / (
                f"{asset_stem}_input_output_{_safe_name(raw_name)}.png"
            )
            _save_input_output_pair(
                pair_input, raw_display, pair_path
            )

        for group_index, (group_name, indices) in enumerate(groups):
            axis = axes[sample_index, group_index + 1]
            if len(indices) == 3:
                response = feature_maps[sample_index, list(indices)].permute(1, 2, 0)
                display = (response / (2.0 * limit) + 0.5).clamp(0, 1).numpy()
                axis.imshow(display)
                summary_kind = "RGB"
            else:
                display = _response_image(
                    feature_maps[sample_index, indices[0]], limit
                )
                axis.imshow(display, cmap="gray", vmin=0, vmax=1)
                summary_kind = "Gray"
            if sample_index == 0:
                axis.set_title(
                    f"{group_name.replace('_', ' ')} {summary_kind}", fontsize=9
                )
            axis.set_xticks([])
            axis.set_yticks([])
            if summary_kind == "RGB":
                rgb_path = class_dir / (
                    f"{asset_stem}_RGB_{_safe_name(group_name)}.png"
                )
                plt.imsave(rgb_path, display)
                _save_nearest_variants(rgb_path)
                pair_path = class_dir / "pairs" / (
                    f"{asset_stem}_input_output_RGB_"
                    f"{_safe_name(group_name)}.png"
                )
                _save_input_output_pair(
                    inputs[sample_index].permute(1, 2, 0).numpy(),
                    display,
                    pair_path,
                )

    figure.suptitle(title)
    figure.subplots_adjust(top=0.9, hspace=0.12, wspace=0.08)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    _save_nearest_variants(path)


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
