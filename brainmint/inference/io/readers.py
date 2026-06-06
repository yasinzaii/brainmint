from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from brainmint.inference.io.base import PathLike, ReaderBase


@dataclass
class NpyReader(ReaderBase):
    """Read arrays saved as .npy."""

    allow_pickle: bool = False

    def read(self, path: PathLike, *, meta: Optional[Mapping[str, Any]] = None) -> np.ndarray:  # noqa: ARG002
        p = Path(path)
        return np.load(str(p), allow_pickle=self.allow_pickle)


@dataclass
class NiftiReader(ReaderBase):
    """Read NIfTI volumes (.nii / .nii.gz) using nibabel."""

    dtype: Any = np.float32

    def read(self, path: PathLike, *, meta: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        p = Path(path)
        try:
            import nibabel as nib  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError("nibabel is required to read NIfTI files.") from e

        img = nib.load(str(p))
        data = img.get_fdata(dtype=self.dtype)
        if meta is not None:
            meta["affine"] = img.affine
            meta["header"] = img.header
        return data


@dataclass
class AutoReader(ReaderBase):
    """Choose a reader based on file extension (.nii/.nii.gz/.npy)."""

    nifti_reader: NiftiReader = field(default_factory=NiftiReader)
    npy_reader: NpyReader = field(default_factory=NpyReader)

    def read(self, path: PathLike, *, meta: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        p = Path(path)
        s = str(p).lower()
        if s.endswith(".nii") or s.endswith(".nii.gz"):
            return self.nifti_reader.read(p, meta=meta)
        return self.npy_reader.read(p, meta=meta)
