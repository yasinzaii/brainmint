from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import LatentInput
from brainmint.inference.diffusion.latent.base import LatentProviderBase


class LatentFromImageShapeDownsample(LatentProviderBase):
    """Derive latent shape from an image tensor in the batch.

    Useful when:
      - the batch contains images but not latents,
      - you want inference to work from image-only batches (Mode-B).

    It does **not** encode the image; it only uses its spatial dims.

    Assumes image tensors are 5D (B,C,Z,Y,X).
    """

    def __init__(
        self,
        *,
        image_key: str = "image",
        latent_channels: int,
        downsample: int = 8,
        require_divisible: bool = True,
    ) -> None:
        super().__init__()
        self.image_key = str(image_key)
        self.latent_channels = int(latent_channels)
        self.downsample = int(downsample)
        self.require_divisible = bool(require_divisible)

    def get_latent(self, batch: Mapping[str, Any], *, ctx: InferenceContext) -> LatentInput:
        if self.image_key not in batch:
            raise KeyError(f"Batch missing image key '{self.image_key}'")
        x = batch[self.image_key]
        if not torch.is_tensor(x) or x.ndim != 5:
            raise ValueError(
                f"Batch[{self.image_key!r}] expected 5D torch.Tensor, got {type(x)} shape={getattr(x,'shape',None)}"
            )
        b = int(x.shape[0])
        spatial = tuple(int(s) for s in x.shape[-3:])
        if self.require_divisible and any(s % self.downsample != 0 for s in spatial):
            raise ValueError(f"Image spatial {spatial} not divisible by downsample={self.downsample}")
        latent_spatial = tuple(int(s // self.downsample) for s in spatial)
        ref = torch.zeros((b, self.latent_channels, *latent_spatial), device=ctx.device, dtype=ctx.dtype)
        return LatentInput(ref=ref)
