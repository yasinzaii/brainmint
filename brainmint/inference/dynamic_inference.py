"""Dynamic inference dispatch for optional inferer-based execution."""

from __future__ import annotations

import math
from typing import Any

import torch


def dynamic_infer(inferer: Any, model: Any, images: torch.Tensor) -> Any:
    """Run a model directly or through an inferer based on input size."""

    if not inferer:
        return model(images)

    if not hasattr(inferer, "roi_size"):
        return inferer(inputs=images, network=model)

    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model(images)

    spatial_dims = images.shape[2:]
    original_roi = inferer.roi_size
    if len(original_roi) != len(spatial_dims):
        raise ValueError(f"ROI length ({len(original_roi)}) does not match spatial dimensions ({len(spatial_dims)}).")

    inferer.roi_size = [min(roi, size) for roi, size in zip(original_roi, spatial_dims, strict=True)]
    try:
        return inferer(network=model, inputs=images)
    finally:
        inferer.roi_size = original_roi

