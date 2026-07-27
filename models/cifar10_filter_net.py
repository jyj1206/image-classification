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
        input_channels=3,
        use_batchnorm=True,
    ):
        super().__init__()
        spatial_channels = int(spatial_kernels)
        input_channels = int(input_channels)
        multiplier = spatial_channels // input_channels
        if input_channels not in (1, 3) or spatial_channels % input_channels:
            raise ValueError("input_channels must be 1 or 3 and divide spatial_kernels")
        if multiplier not in (1, 2, 4):
            raise ValueError("Depthwise kernel multiplier must be 1, 2, or 4")
        expected_multiplier = {"sobel": 2, "mixed": 4}.get(stem_filter, 1)
        if stem_filter != "learnable" and multiplier != expected_multiplier:
            raise ValueError(
                f"{stem_filter} requires {expected_multiplier} kernel(s) per input channel"
            )
        self.use_batchnorm = bool(use_batchnorm)
        self.stem_conv = nn.Conv2d(
            input_channels, spatial_channels, kernel_size=3, stride=1,
            padding=1, groups=input_channels, bias=False,
        )
        # This projection and every layer below it are identical and learnable
        # in all nine experiments.
        self.stem_projection = nn.Conv2d(
            spatial_channels, 64, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.stem_bn = (
            nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        )
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
        layers = [
            ResBlock(
                self.in_channels, channels, stride,
                use_batchnorm=self.use_batchnorm,
            )
        ]
        self.in_channels = channels
        layers.extend(
            ResBlock(channels, channels, use_batchnorm=self.use_batchnorm)
            for _ in range(1, blocks)
        )
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
