from __future__ import annotations

from typing import Any, Dict, Literal

import torch


def build_med_ddpm_mask_batch(
    *,
    batch_size: int = 1,
    modality: str = "T1w",
    condition_key: str = "med_ddpm_mask_one_hot",
    modality_key: str = "modality",
    image_size: int = 192,
    depth_size: int = 144,
    mask_mode: Literal["brain_ellipsoid", "full_brain"] = "brain_ellipsoid",
) -> Dict[str, Any]:
    """Build a no-BrainScape Med-DDPM inference batch.

    Med-DDPM is segmentation-mask conditioned. This helper creates a simple
    one-hot semantic mask with only the brain-area channel active, which is
    useful for smoke-test generation without touching BrainScape datasets.

    The project Med-DDPM wrapper transposes the condition before calling the
    upstream sampler, so this generic request uses pre-upstream spatial order
    ``(image_size, image_size, depth_size)``. After the wrapper transpose, the
    upstream condition matches ``(depth_size, image_size, image_size)``.
    """

    batch = int(batch_size)
    depth = int(depth_size)
    size = int(image_size)

    condition = torch.zeros((batch, 4, size, size, depth), dtype=torch.float32)
    if mask_mode == "full_brain":
        condition[:, 3] = 1.0
    elif mask_mode == "brain_ellipsoid":
        y = torch.linspace(-1.0, 1.0, size).view(size, 1, 1)
        x = torch.linspace(-1.0, 1.0, size).view(1, size, 1)
        z = torch.linspace(-1.0, 1.0, depth).view(1, 1, depth)
        mask = (z / 0.92).square() + (y / 0.72).square() + (x / 0.72).square() <= 1.0
        condition[:, 3] = mask.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported Med-DDPM mask_mode={mask_mode!r}")

    return {
        "batch_size": batch,
        modality_key: [str(modality)] * batch,
        condition_key: condition,
    }
