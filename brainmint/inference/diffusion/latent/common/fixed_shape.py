from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import LatentInput
from brainmint.inference.diffusion.latent.base import LatentProviderBase


class FixedShapeLatent(LatentProviderBase):
    """Create a latent reference tensor from explicit ``(B, C, Z, Y, X)`` parameters."""

    def __init__(
        self,
        *,
        latent_channels: int,
        spatial: Sequence[int],
        batch_key: str | None = "image",
    ) -> None:
        super().__init__()

        self.latent_channels = int(latent_channels)
        if len(spatial) != 3:
            raise ValueError("spatial must be (Z, Y, X)")
        self.spatial = tuple(int(size) for size in spatial)

        if batch_key is None:
            raise ValueError("batch_key must not be None (batch-derived batch size is required).")
        self.batch_key = str(batch_key)

    def _resolve_batch_size(self, batch: Mapping[str, Any]) -> int:
        if self.batch_key not in batch:
            raise KeyError(f"Batch missing '{self.batch_key}' to resolve batch size")

        value = batch[self.batch_key]
        if torch.is_tensor(value):
            batch_size = int(value.item()) if value.ndim == 0 or value.numel() == 1 else int(value.shape[0])
        else:
            batch_size = int(value)

        if batch_size <= 0:
            raise ValueError(f"Resolved batch size must be > 0, got {batch_size}")
        return batch_size

    def get_latent(self, batch: Mapping[str, Any], *, ctx: InferenceContext) -> LatentInput:
        batch_size = self._resolve_batch_size(batch)
        ref = torch.zeros(
            (batch_size, self.latent_channels, *self.spatial),
            device=ctx.device,
            dtype=ctx.dtype,
        )
        return LatentInput(ref=ref)
