"""Non-linear depthwise spatial operations used by the plain-CNN experiments."""

import torch
import torch.nn.functional as F
from torch import nn

from .filters import get_filter_kernels


class SobelMagnitude(nn.Module):
    """One Sobel-magnitude output per input channel using trainable X/Y kernels."""

    def __init__(self, input_channels, trainable=False, eps=1e-12):
        super().__init__()
        self.in_channels = int(input_channels)
        self.out_channels = self.in_channels
        self.groups = self.in_channels
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(2 * self.in_channels, 1, 3, 3))
        kernels = get_filter_kernels("sobel", dtype=self.weight.dtype)
        with torch.no_grad():
            for channel in range(self.in_channels):
                self.weight[2 * channel].copy_(kernels[0])
                self.weight[2 * channel + 1].copy_(kernels[1])
        self.weight.requires_grad_(bool(trainable))

    def forward(self, x):
        responses = F.conv2d(x, self.weight, padding=1, groups=self.groups)
        batch, _, height, width = responses.shape
        responses = responses.reshape(batch, self.in_channels, 2, height, width)
        return torch.sqrt(responses.square().sum(dim=2) + self.eps)


class MixedMagnitude(nn.Module):
    """Identity, Laplacian and Sobel-magnitude outputs per input channel."""

    def __init__(self, input_channels, trainable=False, eps=1e-12):
        super().__init__()
        self.in_channels = int(input_channels)
        self.out_channels = 3 * self.in_channels
        self.groups = self.in_channels
        self.eps = float(eps)
        # Per input group: Identity, Laplacian, Sobel-X, Sobel-Y.
        self.weight = nn.Parameter(torch.empty(4 * self.in_channels, 1, 3, 3))
        identity = get_filter_kernels("identity", dtype=self.weight.dtype)[0]
        laplacian = get_filter_kernels("laplacian", dtype=self.weight.dtype)[0]
        sobel = get_filter_kernels("sobel", dtype=self.weight.dtype)
        with torch.no_grad():
            for channel in range(self.in_channels):
                start = 4 * channel
                self.weight[start].copy_(identity)
                self.weight[start + 1].copy_(laplacian)
                self.weight[start + 2].copy_(sobel[0])
                self.weight[start + 3].copy_(sobel[1])
        self.weight.requires_grad_(bool(trainable))

    def forward(self, x):
        responses = F.conv2d(x, self.weight, padding=1, groups=self.groups)
        batch, _, height, width = responses.shape
        responses = responses.reshape(batch, self.in_channels, 4, height, width)
        magnitude = torch.sqrt(
            responses[:, :, 2].square() + responses[:, :, 3].square() + self.eps
        )
        outputs = torch.stack(
            (responses[:, :, 0], responses[:, :, 1], magnitude), dim=2
        )
        return outputs.reshape(batch, 3 * self.in_channels, height, width)
