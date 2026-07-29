import torch
from torch import nn

from .modules import MixedMagnitude, SobelMagnitude, initialize_stem_filter


class CIFAR10PlainFilterNet(nn.Module):
    """Plain CIFAR CNN with one configurable input spatial operation."""

    def __init__(
        self,
        num_classes=10,
        stem_filter="learnable",
        stem_trainable=True,
        spatial_kernels=3,
        input_channels=3,
        use_batchnorm=False,
    ):
        super().__init__()
        if use_batchnorm:
            raise ValueError("CIFAR10PlainFilterNet is defined without BatchNorm")
        input_channels = int(input_channels)
        spatial_kernels = int(spatial_kernels)
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3")

        if stem_filter == "sobel_magnitude":
            if spatial_kernels != input_channels:
                raise ValueError("Sobel magnitude produces one output per input channel")
            self.stem_conv = SobelMagnitude(input_channels, stem_trainable)
        elif stem_filter == "mixed_magnitude":
            if spatial_kernels != 3 * input_channels:
                raise ValueError("Mixed magnitude produces three outputs per input channel")
            self.stem_conv = MixedMagnitude(input_channels, stem_trainable)
        else:
            if spatial_kernels % input_channels:
                raise ValueError("spatial_kernels must be divisible by input_channels")
            multiplier = spatial_kernels // input_channels
            if multiplier not in (1, 3):
                raise ValueError("Plain-CNN depthwise multiplier must be 1 or 3")
            self.stem_conv = nn.Conv2d(
                input_channels,
                spatial_kernels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=input_channels,
                bias=False,
            )

        self.stem_projection = nn.Conv2d(
            spatial_kernels, 64, kernel_size=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.features = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

        self._initialize_learnable_layers()
        if isinstance(self.stem_conv, nn.Conv2d):
            initialize_stem_filter(
                self.stem_conv, stem_filter, stem_trainable
            )

    def _initialize_learnable_layers(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.relu(self.stem_projection(x))
        x = self.features(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))
