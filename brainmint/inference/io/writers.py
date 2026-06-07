from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from brainmint.inference.io.base import WriterBase

PathLike = str | Path


@dataclass
class NpyWriter(WriterBase):
    """Write arrays as .npy."""

    allow_pickle: bool = False

    def write(self, array: np.ndarray, path: PathLike, *, meta: Mapping[str, Any] | None = None) -> Path:  # noqa: ARG002
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), array, allow_pickle=self.allow_pickle)
        return p


@dataclass
class NiftiWriter(WriterBase):
    """Write 3D volumes as NIfTI (.nii / .nii.gz) using nibabel.

    If nibabel is not installed, this writer raises an ImportError.
    """

    affine: np.ndarray | None = None
    dtype: Any = np.float32

    def write(self, array: np.ndarray, path: PathLike, *, meta: Mapping[str, Any] | None = None) -> Path:  # noqa: ARG002
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        try:
            import nibabel as nib  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "nibabel is required to write NIfTI files. Install it or use NpyWriter/VolumeWriter fallback."
            ) from e

        aff = self.affine
        if aff is None:
            aff = np.eye(4, dtype=np.float32)

        # Common shapes:
        # - (H,W,D)
        # - (C,H,W,D) -> write each channel separately in caller, or pass a single channel here.
        vol = np.asarray(array).astype(self.dtype, copy=False)
        nii = nib.Nifti1Image(vol, affine=aff)
        nib.save(nii, str(p))
        return p


@dataclass
class PngWriter(WriterBase):
    """Write 2D arrays as PNG using Pillow.

    For non-image tensors, this writer applies a simple min-max normalization to [0,255]
    and writes an 8-bit grayscale image.
    """

    normalize: bool = True

    def write(self, array: np.ndarray, path: PathLike, *, meta: Mapping[str, Any] | None = None) -> Path:  # noqa: ARG002
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        try:
            from PIL import Image  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "Pillow is required to write PNG files. "
                "Install it with `pip install brainmint[io]` or use NpyWriter/VolumeWriter."
            ) from e

        img = np.asarray(array)
        if img.ndim != 2:
            raise ValueError(f"PngWriter expects a 2D array, got shape={img.shape}")

        if self.normalize:
            x = img.astype(np.float32)
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
            if hi > lo:
                x = (x - lo) / (hi - lo)
            else:
                x = x * 0.0
            img8 = (x * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img8 = img.astype(np.uint8, copy=False)

        Image.fromarray(img8, mode="L").save(str(p))
        return p


@dataclass
class VolumeWriter(WriterBase):
    """Write a 3D volume, preferring NIfTI when requested by extension.

    - If path ends with .nii or .nii.gz: tries NIfTI via nibabel.
      If nibabel is unavailable and fallback_to_npy=True, writes .npy instead.
    - Otherwise: writes .npy.
    """

    fallback_to_npy: bool = True
    nifti_affine: np.ndarray | None = None

    def write(self, array: np.ndarray, path: PathLike, *, meta: Mapping[str, Any] | None = None) -> Path:  # noqa: ARG002
        p = Path(path)
        p_str = str(p).lower()

        if p_str.endswith(".nii") or p_str.endswith(".nii.gz"):
            try:
                return NiftiWriter(affine=self.nifti_affine).write(array, p)
            except ImportError:
                if not self.fallback_to_npy:
                    raise
                # For .nii.gz we want to drop both suffixes -> .npy
                if p_str.endswith(".nii.gz"):
                    npy_path = p.with_name(p.name[:-7] + ".npy")
                else:
                    npy_path = p.with_suffix(".npy")
                return NpyWriter().write(array, npy_path)

        return NpyWriter().write(array, p)

    def write_4d(
        self,
        array: np.ndarray,
        path: PathLike,
        *,
        channels_first: bool = True,
        meta: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write a 4D latent, preserving channel as 4th dim in NIfTI.

        Expected:
          - channels_first=True : (C, Z, Y, X) -> saved as (Z, Y, X, C)
          - channels_first=False: (Z, Y, X, C) -> saved as-is
        """
        arr = np.asarray(array)
        if arr.ndim != 4:
            raise ValueError(f"VolumeWriter.write_4d expects 4D array, got shape={arr.shape}")

        if channels_first:
            arr = np.moveaxis(arr, 0, -1)  # (C,Z,Y,X) -> (Z,Y,X,C)

        # NIfTI interop is simplest as float32
        arr = arr.astype(np.float32, copy=False)

        p = Path(path)
        p_str = str(p).lower()

        if p_str.endswith(".nii") or p_str.endswith(".nii.gz"):
            try:
                return NiftiWriter(affine=self.nifti_affine, dtype=np.float32).write(arr, p, meta=meta)
            except ImportError:
                if not self.fallback_to_npy:
                    raise
                # For .nii.gz we want to drop both suffixes -> .npy
                if p_str.endswith(".nii.gz"):
                    npy_path = p.with_name(p.name[:-7] + ".npy")
                else:
                    npy_path = p.with_suffix(".npy")
                return NpyWriter().write(arr, npy_path, meta=meta)

        return NpyWriter().write(arr, p, meta=meta)


@dataclass
class AutoWriter(WriterBase):
    """Choose a writer based on file extension (.nii/.nii.gz/.npy/.png)."""

    # dataclasses require default_factory for mutable default instances
    volume_writer: VolumeWriter = field(default_factory=VolumeWriter)
    png_writer: PngWriter = field(default_factory=PngWriter)

    def write(self, array: np.ndarray, path: PathLike, *, meta: Mapping[str, Any] | None = None) -> Path:
        p = Path(path)
        s = str(p).lower()
        if s.endswith(".png"):
            return self.png_writer.write(array, p, meta=meta)
        # For npy and nifti, VolumeWriter already handles both
        return self.volume_writer.write(array, p, meta=meta)
