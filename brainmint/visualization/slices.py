from __future__ import annotations

import numpy as np


def axis_for_orientation(orientation: str) -> int:
    """
    Keep consistent with existing repo scripts:
      sagittal -> axis 0
      coronal  -> axis 1
      axial    -> axis 2
    Where the volume is treated as [D, H, W].
    """
    o = orientation.lower()
    if o == "sagittal":
        return 0
    if o == "coronal":
        return 1
    if o == "axial":
        return 2
    raise ValueError(f"Unknown slice_orientation: {orientation!r}")


def normalize_to_uint8(
    data: np.ndarray,
    *,
    upper_percentile: float = 99.5,
    lower_percentile: float = 0.5,
) -> np.ndarray:
    """Normalise a 2D array to uint8 with percentile clipping."""
    if data.size == 0:
        return np.zeros_like(data, dtype=np.uint8)

    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros_like(data, dtype=np.uint8)

    vals = data[finite]
    lo = float(vals.min()) if lower_percentile <= 0.0 else float(np.percentile(vals, lower_percentile))
    hi = float(np.percentile(vals, upper_percentile))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.uint8)

    scaled = (data - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)
