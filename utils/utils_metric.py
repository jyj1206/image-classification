import torch


def psnr(pred, target):
    from torchmetrics.functional.image import peak_signal_noise_ratio

    pred_y = _rgb_to_y(_to_nchw(pred))
    target_y = _rgb_to_y(_to_nchw(target)).to(device=pred_y.device, dtype=pred_y.dtype)

    return peak_signal_noise_ratio(
        pred_y,
        target_y,
        data_range=1.0,
        reduction="elementwise_mean",
        dim=(1, 2, 3),
    ).item()


def ssim(pred, target):
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    pred_y = _rgb_to_y(_to_nchw(pred))
    target_y = _rgb_to_y(_to_nchw(target)).to(device=pred_y.device, dtype=pred_y.dtype)

    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred_y.device)
    return metric(pred_y, target_y).item()


def lpips(pred, target, metric=None, net_type="alex"):
    pred = _to_nchw(pred)
    target = _to_nchw(target).to(device=pred.device, dtype=pred.dtype)

    if metric is None:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type,
            normalize=True,
        ).to(pred.device)

    metric.eval()
    with torch.no_grad():
        return metric(pred, target).item()


def final_score(psnr_value, ssim_value, lpips_value):
    return psnr_value + 10 * ssim_value - 5 * lpips_value


def final_score_from_images(pred, target, lpips_metric=None):
    psnr_value = psnr(pred, target)
    ssim_value = ssim(pred, target)
    lpips_value = lpips(pred, target, metric=lpips_metric)

    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
        "lpips": lpips_value,
        "final_score": final_score(psnr_value, ssim_value, lpips_value),
    }


def _to_nchw(image):
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    image = image.float()
    if image.max() > 2:
        image = image / 255.0

    if image.ndim == 3:
        if image.shape[0] in (1, 3):
            image = image.unsqueeze(0)
        else:
            image = image.permute(2, 0, 1).unsqueeze(0)
    elif image.ndim == 4 and image.shape[-1] in (1, 3):
        image = image.permute(0, 3, 1, 2)

    return image.contiguous().clamp(0, 1)


def _rgb_to_y(image):
    if image.size(1) == 1:
        return image

    if image.size(1) != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.size(1)}")

    r = image[:, 0:1]
    g = image[:, 1:2]
    b = image[:, 2:3]
    return 0.256789 * r + 0.504129 * g + 0.097906 * b + 16.0 / 255.0
