from __future__ import annotations

from typing import Any

import torch
from torch import nn

from brainmint.models.blocks.haar_dwt import (
    HaarWaveletTransform3D,
    InverseHaarWaveletTransform3D,
    TwoStageInverseWaveletTransform,
    TwoStageWaveletTransform,
)


class DWTCompressionModule(nn.Module):
    """Analytic Haar-DWT compressor with paired inverse reconstruction."""

    def __init__(self, *, stages: int = 1, in_channels: int | None = None) -> None:
        super().__init__()
        if stages not in (1, 2):
            raise ValueError("Only one- and two-stage DWT compression are supported.")

        self.stages = int(stages)
        self.in_channels = in_channels
        self.spatial_compression = 2**self.stages
        self.latent_channels = None if in_channels is None else int(in_channels) * (8**self.stages)

        if self.stages == 1:
            self.forward_transform: nn.Module = HaarWaveletTransform3D()
            self.inverse_transform: nn.Module = InverseHaarWaveletTransform3D()
        else:
            if in_channels is None:
                raise ValueError("in_channels must be provided for two-stage DWT compression.")
            self.forward_transform = TwoStageWaveletTransform(in_channels=in_channels)
            self.inverse_transform = TwoStageInverseWaveletTransform(out_channels=in_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_transform(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.inverse_transform(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def run_inference(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None, None]:
        return self.reconstruct(batch["image"]), None, None

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        return self.reconstruct(x)

