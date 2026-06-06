import torch


def adaptive_background_mask(
    image: torch.Tensor,
    quantile: float = 0.10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute an adaptive background mask using a per-volume percentile cutoff.

    Args:
        image: Tensor with shape [D, H, W], [C, D, H, W], or [B, C, D, H, W].
        quantile: Quantile used for the cutoff (default: 0.10).
        eps: Small value added to the threshold before masking.

    Returns:
        Boolean mask with the same shape as ``image``.
    """
    if image.ndim not in (3, 4, 5):
        raise ValueError(
            "Expected volume with 3D, 4D, or 5D shape, "
            f"but got shape {tuple(image.shape)}."
        )

    if image.ndim == 5:
        batch_size = image.shape[0]
        flat = image.reshape(batch_size, -1)
        thresholds = torch.quantile(flat, quantile, dim=1, keepdim=True)
        thresholds = thresholds.view(batch_size, *([1] * (image.ndim - 1)))
    else:
        flat = image.reshape(1, -1)
        thresholds = torch.quantile(flat, quantile, dim=1, keepdim=True)
        thresholds = thresholds.squeeze()

    return image > (thresholds + eps)
