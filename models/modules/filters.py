"""Deterministic initializers for the single experimental CIFAR-10 stem."""

import torch
from torch import nn


FILTER_TYPES = ("learnable", "identity", "sobel", "laplacian", "gaussian", "mixed")


def get_filter_kernels(filter_type, *, device=None, dtype=None):
    """Return normalized 3x3 kernels as a tensor of shape (K, 3, 3)."""
    dtype = dtype or torch.float32
    kernels = {
        "identity": [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
        "sobel": [
            [[-1 / 8, 0, 1 / 8], [-2 / 8, 0, 2 / 8], [-1 / 8, 0, 1 / 8]],
            [[-1 / 8, -2 / 8, -1 / 8], [0, 0, 0], [1 / 8, 2 / 8, 1 / 8]],
        ],
        "laplacian": [[[0, 1 / 8, 0], [1 / 8, -4 / 8, 1 / 8], [0, 1 / 8, 0]]],
        "gaussian": [[[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]]],
        "mixed": [
            [[-1 / 8, 0, 1 / 8], [-2 / 8, 0, 2 / 8], [-1 / 8, 0, 1 / 8]],
            [[-1 / 8, -2 / 8, -1 / 8], [0, 0, 0], [1 / 8, 2 / 8, 1 / 8]],
            [[0, 1 / 8, 0], [1 / 8, -4 / 8, 1 / 8], [0, 1 / 8, 0]],
            [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]],
        ],
    }
    if filter_type not in kernels:
        raise ValueError(f"No handcrafted kernel for filter_type={filter_type!r}")
    return torch.tensor(kernels[filter_type], device=device, dtype=dtype)


def initialize_stem_filter(conv, filter_type="learnable", trainable=True):
    """Initialize the depthwise spatial operation for one experiment."""
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"filter_type must be one of {FILTER_TYPES}, got {filter_type!r}")
    input_channels = conv.in_channels
    expected_multiplier = {
        "identity": 1,
        "laplacian": 1,
        "gaussian": 1,
        "sobel": 2,
        "mixed": 4,
    }.get(filter_type)
    multiplier = conv.out_channels // input_channels
    valid_multiplier = multiplier in (1, 2, 3, 4) if filter_type == "learnable" else multiplier == expected_multiplier
    expected_shape = (conv.out_channels, 1, 3, 3)
    if (
        conv.out_channels % input_channels != 0
        or not valid_multiplier
        or tuple(conv.weight.shape) != expected_shape
        or conv.groups != input_channels
    ):
        raise ValueError(
            f"Invalid {filter_type} depthwise stem: in={input_channels}, "
            f"out={conv.out_channels}, groups={conv.groups}, "
            f"weight={tuple(conv.weight.shape)}"
        )

    with torch.no_grad():
        if filter_type == "learnable":
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        else:
            kernels = get_filter_kernels(filter_type, device=conv.weight.device, dtype=conv.weight.dtype)
            conv.weight.zero_()
            if filter_type == "mixed":
                # Per RGB group: Sobel-X, Sobel-Y, Laplacian, Gaussian.
                for input_channel in range(input_channels):
                    start = 4 * input_channel
                    for kernel_idx in range(4):
                        conv.weight[start + kernel_idx, 0].copy_(kernels[kernel_idx])
            elif filter_type == "sobel":
                # Grouped-convolution output order:
                # R-X, R-Y, G-X, G-Y, B-X, B-Y.
                for input_channel in range(input_channels):
                    conv.weight[2 * input_channel, 0].copy_(kernels[0])
                    conv.weight[2 * input_channel + 1, 0].copy_(kernels[1])
            else:
                for input_channel in range(input_channels):
                    conv.weight[input_channel, 0].copy_(kernels[0])

    conv.weight.requires_grad_(bool(trainable))
    return conv
