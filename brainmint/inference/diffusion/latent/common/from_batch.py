from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import LatentInput
from brainmint.inference.diffusion.latent.base import LatentProviderBase


class LatentFromBatch(LatentProviderBase):
    """Use a latent tensor already present in the batch as the reference shape."""

    def __init__(self, *, key: str = "latent", ensure_5d: bool = True) -> None:
        super().__init__()
        self.key = str(key)
        self.ensure_5d = bool(ensure_5d)

    def get_latent(self, batch: Mapping[str, Any], *, ctx: InferenceContext) -> LatentInput:
        if self.key not in batch:
            raise KeyError(f"Batch missing latent key '{self.key}'")
        latent = batch[self.key]
        if not torch.is_tensor(latent):
            raise TypeError(f"Batch[{self.key!r}] must be torch.Tensor, got {type(latent)}")
        if self.ensure_5d and latent.ndim != 5:
            raise ValueError(f"Batch[{self.key!r}] must be 5D (B,C,Z,Y,X), got {tuple(latent.shape)}")

        ref = torch.zeros_like(latent.to(device=ctx.device, dtype=ctx.dtype))
        return LatentInput(ref=ref)
