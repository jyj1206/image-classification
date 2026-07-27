import torch
from torch import nn

from .modules import ResBlock, initialize_stem_filter


class CIFAR10FilterNet(nn.Module):
    """CIFAR classifier whose only experimental component is ``stem_conv``."""

    def __init__(
        self,
        num_classes=10,
        stem_filter="learnable",
        stem_trainable=True,
        spatial_kernels=3,
    ):
        super().__init__()
        spatial_channels = int(spatial_kernels)
        if spatial_channels not in (3, 6, 12):
            raise ValueError("spatial_kernels must be 3, 6, or 12")
        if stem_filter == "sobel" and spatial_channels != 6:
            raise ValueError("Sobel requires spatial_kernels=6")
        if stem_filter == "mixed" and spatial_channels != 12:
            raise ValueError("Mixed filters require spatial_kernels=12")
        if stem_filter not in ("learnable", "sobel", "mixed") and spatial_channels != 3:
            raise ValueError(f"{stem_filter} requires spatial_kernels=3")
        self.stem_conv = nn.Conv2d(
            3, spatial_channels, kernel_size=3, stride=1,
            padding=1, groups=3, bias=False,
        )
        # This projection and every layer below it are identical and learnable
        # in all nine experiments.
        self.stem_projection = nn.Conv2d(
            spatial_channels, 64, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = 64
        self.stage1 = self._make_stage(64, blocks=2, stride=1)
        self.stage2 = self._make_stage(128, blocks=2, stride=2)
        self.stage3 = self._make_stage(256, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

        self._initialize_common_layers()
        initialize_stem_filter(self.stem_conv, stem_filter, stem_trainable)

    def _make_stage(self, channels, blocks, stride):
        layers = [ResBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(ResBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_common_layers(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.relu(self.stem_bn(self.stem_projection(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))
