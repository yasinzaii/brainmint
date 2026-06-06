from __future__ import annotations

"""BraSyn / BrainLesion MissingMRI NIfTI and tensor IO helpers."""

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from brainmint.inference.io import NiftiReader, NiftiWriter


def ensure_b1hwd(x: torch.Tensor) -> torch.Tensor:
    """Ensure a volume tensor is shaped ``(B, 1, H, W, D)``."""

    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if x.ndim == 3:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 4:
        if x.shape[0] == 1:
            return x.unsqueeze(0)
        raise ValueError(f"Ambiguous 4D tensor shape {tuple(x.shape)}; expected (1,H,W,D)")
    if x.ndim == 5:
        return x
    raise ValueError(f"Expected 3D/4D/5D tensor, got shape {tuple(x.shape)}")


def center_crop_or_pad(x: torch.Tensor, target_hwd: Sequence[int]) -> torch.Tensor:
    """Center crop or zero-pad a ``(B, 1, H, W, D)`` tensor."""

    x = ensure_b1hwd(x)
    _, _, h, w, d = x.shape
    th, tw, td = (int(v) for v in target_hwd)

    if h > th:
        h0 = (h - th) // 2
        x = x[:, :, h0 : h0 + th, :, :]
        h = th
    if w > tw:
        w0 = (w - tw) // 2
        x = x[:, :, :, w0 : w0 + tw, :]
        w = tw
    if d > td:
        d0 = (d - td) // 2
        x = x[:, :, :, :, d0 : d0 + td]
        d = td

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


def _read_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {}
    data = NiftiReader(dtype=np.float32).read(path, meta=meta)
    return np.asarray(data, dtype=np.float32), meta


def _write_nifti(data: np.ndarray, path: str | Path, *, affine: Optional[np.ndarray] = None) -> None:
    NiftiWriter(affine=affine, dtype=np.float32).write(np.asarray(data, dtype=np.float32), path)


def _extract_filename_from_tensor(t: torch.Tensor) -> Optional[str]:
    """Best-effort extraction of the original filename from a MONAI MetaTensor."""

    meta = getattr(t, "meta", None) or {}
    filename = meta.get("filename_or_obj") or meta.get("original_filename_or_obj")
    if isinstance(filename, (str, Path)):
        return str(filename)
    if isinstance(filename, (list, tuple)) and filename and isinstance(filename[0], (str, Path)):
        return str(filename[0])
    return None


def _affine_from_tensor(t: torch.Tensor) -> Optional[np.ndarray]:
    """Best-effort extraction of an affine matrix from a MONAI MetaTensor."""

    meta = getattr(t, "meta", None) or {}
    affine = meta.get("affine") or meta.get("original_affine")
    if affine is None:
        return None
    try:
        return np.asarray(affine, dtype=np.float32)
    except Exception:
        return None


def _create_zero(shape_hwd: Sequence[int], out_path: str | Path, *, affine: Optional[np.ndarray] = None) -> None:
    """Create a zero-valued NIfTI with the requested shape and affine."""

    zero = np.zeros(tuple(int(x) for x in shape_hwd), dtype=np.float32)
    _write_nifti(zero, out_path, affine=affine)


def _tensor_to_nifti(x_b1hwd: torch.Tensor, out_path: str | Path, *, affine: Optional[np.ndarray] = None) -> None:
    """Save a ``(B, 1, H, W, D)`` tensor as a NIfTI volume."""

    x_b1hwd = ensure_b1hwd(x_b1hwd).detach().cpu()
    volume = x_b1hwd[0, 0].numpy().astype(np.float32)
    _write_nifti(volume, out_path, affine=affine)


def _create_zero_like(reference_path: str | Path, out_path: str | Path) -> None:
    """Create a zero-valued NIfTI with the same shape and affine as a reference."""

    data, meta = _read_nifti(reference_path)
    _write_nifti(np.zeros_like(data, dtype=np.float32), out_path, affine=meta.get("affine"))


def _tensor_to_nifti_like(x_b1hwd: torch.Tensor, reference_path: str | Path, out_path: str | Path) -> None:
    """Save a tensor as NIfTI using affine/header metadata from a reference path."""

    x_b1hwd = ensure_b1hwd(x_b1hwd).detach().cpu()
    volume = x_b1hwd[0, 0].numpy().astype(np.float32)
    _, meta = _read_nifti(reference_path)
    _write_nifti(volume, out_path, affine=meta.get("affine"))


def _nifti_to_tensor(path: str | Path) -> torch.Tensor:
    data, _ = _read_nifti(path)
    return torch.from_numpy(data).unsqueeze(0).unsqueeze(0)


def load_nifti(path: str | Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a NIfTI volume and affine through BrainMint IO."""

    data, meta = _read_nifti(path)
    return data, meta.get("affine")


def save_nifti(path: str | Path, data: np.ndarray, affine: Optional[np.ndarray]) -> None:
    """Save a NIfTI volume through BrainMint IO."""

    _write_nifti(data, path, affine=affine)


def create_zero_nifti_like(reference_path: str | Path, out_path: str | Path) -> tuple[Path, Optional[np.ndarray]]:
    """Create a zero-valued NIfTI with the same shape and affine as a reference."""

    data, meta = _read_nifti(reference_path)
    affine = meta.get("affine")
    _write_nifti(np.zeros_like(data, dtype=np.float32), out_path, affine=affine)
    return Path(out_path), affine
