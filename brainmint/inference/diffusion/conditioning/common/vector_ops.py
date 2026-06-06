from __future__ import annotations

from typing import Tuple

import torch


def as_2d(x: torch.Tensor, *, name: str = "tensor") -> torch.Tensor:
    """
    Ensure x is (B, C).

    Raises:
      ValueError if ndim != 2.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D (B,C), got shape={tuple(x.shape)}")
    return x


def to_context(vec_bc: torch.Tensor) -> torch.Tensor:
    """(B,C) -> (B,1,C) (single-token context for cross-attn)."""
    vec_bc = as_2d(vec_bc, name="vec_bc")
    return vec_bc.unsqueeze(1)


def expand_to_spatial(vec_bc: torch.Tensor, spatial: Tuple[int, int, int]) -> torch.Tensor:
    """(B,C) -> (B,C,Z,Y,X) by broadcasting."""
    vec_bc = as_2d(vec_bc, name="vec_bc")
    b, c = vec_bc.shape
    z, y, x = (int(s) for s in spatial)
    return vec_bc.view(b, c, 1, 1, 1).expand(b, c, z, y, x)
