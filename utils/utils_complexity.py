"""Deterministic inference-complexity accounting for the CIFAR models."""

import torch
from torch import nn

from models.modules import MixedMagnitude, SobelMagnitude


def analyze_inference_complexity(model, input_shape, device):
    """Count parameters, MACs and forward FLOPs for one input image.

    Convention:
    - one multiply-accumulate (MAC) is reported as two FLOPs;
    - convolution/linear bias, ReLU, average pooling and Sobel-magnitude
      elementwise operations are included in FLOPs;
    - reshape, concatenation and memory movement are excluded.
    """
    totals = {"macs": 0, "extra_flops": 0}
    handles = []

    def conv_hook(module, inputs, output):
        batch, out_channels, out_height, out_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        macs = (
            batch
            * out_channels
            * out_height
            * out_width
            * (module.in_channels // module.groups)
            * kernel_height
            * kernel_width
        )
        totals["macs"] += int(macs)
        if module.bias is not None:
            totals["extra_flops"] += int(output.numel())

    def linear_hook(module, inputs, output):
        batch = output.numel() // module.out_features
        totals["macs"] += int(batch * module.in_features * module.out_features)
        if module.bias is not None:
            totals["extra_flops"] += int(output.numel())

    def relu_hook(module, inputs, output):
        totals["extra_flops"] += int(output.numel())

    def pool_hook(module, inputs, output):
        input_tensor = inputs[0]
        input_spatial = input_tensor.shape[-2] * input_tensor.shape[-1]
        # (N-1) additions plus one division per pooled output.
        totals["extra_flops"] += int(output.numel() * input_spatial)

    def magnitude_hook(module, inputs, output):
        batch, _, out_height, out_width = output.shape
        input_channels = module.in_channels
        internal_kernels = (
            2 * input_channels
            if isinstance(module, SobelMagnitude)
            else 4 * input_channels
        )
        totals["macs"] += int(
            batch * internal_kernels * out_height * out_width * 3 * 3
        )
        # Gx^2, Gy^2, their sum, epsilon addition and sqrt: five operations.
        totals["extra_flops"] += int(
            batch * input_channels * out_height * out_width * 5
        )

    custom_types = (SobelMagnitude, MixedMagnitude)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(relu_hook))
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            handles.append(module.register_forward_hook(pool_hook))
        elif isinstance(module, custom_types):
            handles.append(module.register_forward_hook(magnitude_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            dummy = torch.zeros(input_shape, device=device)
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    macs = totals["macs"]
    flops = 2 * macs + totals["extra_flops"]
    return {
        "complexity_batch_size": int(input_shape[0]),
        "complexity_input_shape": "x".join(str(value) for value in input_shape),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "macs": int(macs),
        "macs_m": macs / 1_000_000,
        "flops": int(flops),
        "flops_m": flops / 1_000_000,
        "flops_convention": (
            "forward only; batch=1; 1 MAC=2 FLOPs; includes bias, ReLU, "
            "average pooling, and Sobel-magnitude elementwise operations"
        ),
    }
