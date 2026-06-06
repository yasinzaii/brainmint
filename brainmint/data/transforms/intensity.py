from __future__ import annotations

import torch
from monai.transforms import MapTransform

from brainmint.data.utils import adaptive_background_mask


class AdaptiveBackgroundZScoreNormalizeD(MapTransform):
    """Z-score normalize an image using a percentile-derived foreground mask.

    The mean/std are computed over the masked voxels only and the background is
    set to zero. This mirrors the metric-time percentile masking but operates
    per sample as a dataset transform.
    """

    def __init__(
        self,
        keys,
        quantile: float = 0.10,
        eps: float = 1e-6,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.quantile = float(quantile)
        self.eps = float(eps)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            if image is None:
                continue
            if not torch.is_tensor(image):
                image = torch.as_tensor(image)

            mask = adaptive_background_mask(image, quantile=self.quantile, eps=self.eps)

            if mask.any():
                masked_values = image[mask]
                mean = masked_values.mean()
                std = masked_values.std(unbiased=False).clamp_min(self.eps)
            else:
                mean = torch.zeros((), device=image.device, dtype=image.dtype)
                std = torch.ones((), device=image.device, dtype=image.dtype)

            normalized = (image - mean) / std
            background = torch.zeros((), device=image.device, dtype=image.dtype)
            d[key] = torch.where(mask, normalized, background)
        return d
