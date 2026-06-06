from __future__ import annotations

"""Small volume/tensor utilities for translation wrappers.

These helpers are intentionally tiny so translation wrappers can remain thin
and mostly delegate to model or integration code.
"""

from typing import Mapping, Optional, Tuple

import torch


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a torch device from config.

    Accepts "auto" (default), a string like "cuda:0", or a torch.device.
    """

    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def ensure_b1hwd(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is shaped (B,1,H,W,D).

    Accepts:
      - (H,W,D)
      - (1,H,W,D)
      - (B,1,H,W,D)
    """

    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if x.ndim == 3:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 4:
        if x.shape[0] == 1:
            # (1,H,W,D)
            return x.unsqueeze(0)
        raise ValueError(f"Ambiguous 4D tensor shape {tuple(x.shape)}; expected (1,H,W,D)")
    if x.ndim == 5:
        return x
    raise ValueError(f"Expected 3D/4D/5D tensor, got shape {tuple(x.shape)}")


def center_crop_or_pad(x: torch.Tensor, target_hwd: Tuple[int, int, int]) -> torch.Tensor:
    """Center crop or zero-pad a (B,1,H,W,D) tensor to target (H,W,D)."""

    x = ensure_b1hwd(x)
    b, c, h, w, d = x.shape
    th, tw, td = map(int, target_hwd)

    # Crop
    if h > th:
        dh = h - th
        h0 = dh // 2
        x = x[:, :, h0 : h0 + th, :, :]
        h = th
    if w > tw:
        dw = w - tw
        w0 = dw // 2
        x = x[:, :, :, w0 : w0 + tw, :]
        w = tw
    if d > td:
        dd = d - td
        d0 = dd // 2
        x = x[:, :, :, :, d0 : d0 + td]
        d = td

    # Pad (pad order for 5D: (D_left,D_right,W_left,W_right,H_left,H_right))
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    pad_d = max(0, td - d)

    if pad_h or pad_w or pad_d:
        ph0 = pad_h // 2
        ph1 = pad_h - ph0
        pw0 = pad_w // 2
        pw1 = pad_w - pw0
        pd0 = pad_d // 2
        pd1 = pad_d - pd0
        x = torch.nn.functional.pad(x, (pd0, pd1, pw0, pw1, ph0, ph1))

    return x


def center_crop_or_pad_hwd(x: torch.Tensor, target_hwd: Tuple[int, int, int]) -> torch.Tensor:
    """Backward-compatible alias."""

    return center_crop_or_pad(x, target_hwd)


def pick_first_tensor(batch: Mapping[str, object], keys: list[str]) -> Optional[torch.Tensor]:
    for k in keys:
        v = batch.get(k)
        if torch.is_tensor(v):
            return v
    return None


def modality_key(modality: str, prefix: str = "real_") -> str:
    return f"{prefix}{str(modality).lower()}"
