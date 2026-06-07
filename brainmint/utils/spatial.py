"""Spatial tensor helpers for inference volumes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def center_crop_or_pad_zyx(x: torch.Tensor, out_size_zyx: Sequence[int]) -> torch.Tensor:
    """Center crop or zero-pad a 5D ``(B, C, Z, Y, X)`` tensor."""

    if x.dim() != 5:
        raise ValueError(f"Expected 5D (B,C,Z,Y,X) tensor, got {tuple(x.shape)}")

    z, y, xw = x.shape[-3:]
    oz, oy, ox = (int(value) for value in out_size_zyx)

    pad_z = max(0, oz - z)
    pad_y = max(0, oy - y)
    pad_x = max(0, ox - xw)

    if pad_z or pad_y or pad_x:
        pad = (
            pad_x // 2,
            pad_x - pad_x // 2,
            pad_y // 2,
            pad_y - pad_y // 2,
            pad_z // 2,
            pad_z - pad_z // 2,
        )
        x = F.pad(x, pad)

    z, y, xw = x.shape[-3:]
    start_z = max(0, (z - oz) // 2)
    start_y = max(0, (y - oy) // 2)
    start_x = max(0, (xw - ox) // 2)

    return x[..., start_z : start_z + oz, start_y : start_y + oy, start_x : start_x + ox]


def center_crop_or_pad_3d(volume: np.ndarray, out_size_zyx: Sequence[int]) -> np.ndarray:
    """Center crop or zero-pad a 3D ``(Z, Y, X)`` array."""

    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X) array, got {arr.shape}")

    z, y, x = arr.shape
    oz, oy, ox = (int(value) for value in out_size_zyx)

    pad_z = max(0, oz - z)
    pad_y = max(0, oy - y)
    pad_x = max(0, ox - x)

    if pad_z or pad_y or pad_x:
        pads = (
            (pad_z // 2, pad_z - pad_z // 2),
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
        )
        arr = np.pad(arr, pads, mode="constant", constant_values=0)

    z, y, x = arr.shape
    start_z = max(0, (z - oz) // 2)
    start_y = max(0, (y - oy) // 2)
    start_x = max(0, (x - ox) // 2)

    return arr[start_z : start_z + oz, start_y : start_y + oy, start_x : start_x + ox]
