from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ....core.context import InferenceContext
from ..base import ConditioningBuilderBase
from .vector_ops import expand_to_spatial, to_context


class RandomVectorConditioning(ConditioningBuilderBase):
    """Random vector conditioning (useful for smoke tests / ablations)."""

    def __init__(self, dim: int, *, key: str = "cond", seed: int | None = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.key = str(key)
        self.seed = seed

    def build(self, batch: Mapping[str, Any], *, latent_ref: torch.Tensor, ctx: InferenceContext) -> dict[str, torch.Tensor]:  # noqa: ARG002
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device=ctx.device)
            gen.manual_seed(int(self.seed))

        b = int(latent_ref.shape[0])
        vec = torch.randn((b, self.dim), device=ctx.device, dtype=ctx.dtype, generator=gen)
        spatial = tuple(int(s) for s in latent_ref.shape[-3:])

        return {
            self.key: vec,
            "context": to_context(vec),
            "cond_concat": expand_to_spatial(vec, spatial),
        }
