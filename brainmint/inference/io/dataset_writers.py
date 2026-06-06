from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch


class SyntheticDatasetWriter:
    """Write generated samples to disk and maintain a JSON index.

    This is intentionally lightweight and script-friendly. It does *not* assume a specific datalist
    schema beyond "records with file paths".

    Typical usage:

    .. code-block:: python

        writer = SyntheticDatasetWriter(out_dir="out/synth", json_path="out/datalist.json")
        for batch in loader:
            out = module.run(batch)
            writer.write(sample=out["sample"], meta={"subject_id": ..., "age": ...})
        writer.flush()

    Notes:
      - Uses ``nibabel`` if available to write NIfTI.
      - Accepts tensors shaped (B,C,Z,Y,X) or (C,Z,Y,X) or (Z,Y,X). Writes one file per item in batch.
      - Stores the path in each record under ``image_key`` (default: "image").
    """

    def __init__(
        self,
        *,
        out_dir: Union[str, Path],
        json_path: Union[str, Path],
        image_key: str = "image",
        filename_template: str = "{index:06d}.nii.gz",
        start_index: int = 0,
        json_indent: int = 2,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = Path(json_path)
        self.image_key = str(image_key)
        self.filename_template = str(filename_template)
        self._index = int(start_index)
        self._records: List[Dict[str, Any]] = []
        self._json_indent = int(json_indent)

    @staticmethod
    def _require_nibabel():
        try:
            import nibabel as nib  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "SyntheticDatasetWriter requires nibabel for NIfTI writing. Install nibabel."
            ) from e
        return nib

    @staticmethod
    def _to_numpy_zyx(x: torch.Tensor) -> np.ndarray:
        # Convert to CPU numpy; squeeze channel dim if C==1.
        x = x.detach().cpu()
        if x.ndim == 5:
            raise ValueError("Expected a single item tensor, not a batched tensor (B,C,Z,Y,X).")
        if x.ndim == 4:
            # (C,Z,Y,X)
            if x.shape[0] == 1:
                x = x[0]
            else:
                # Multi-channel -> keep as (Z,Y,X,C) for NIfTI compatibility
                x = x.permute(1, 2, 3, 0)
        if x.ndim != 3 and x.ndim != 4:
            raise ValueError(f"Expected (Z,Y,X) or (C,Z,Y,X) or (Z,Y,X,C), got {tuple(x.shape)}")
        return x.numpy()

    def write(
        self,
        *,
        sample: torch.Tensor,
        meta: Optional[Mapping[str, Any]] = None,
        affine: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Write sample(s) and return the appended JSON records."""
        nib = self._require_nibabel()
        meta = dict(meta or {})
        affine_arr = affine if affine is not None else np.eye(4, dtype=np.float32)

        if not torch.is_tensor(sample):
            raise TypeError(f"sample must be torch.Tensor, got {type(sample)}")

        # Normalize to (B, ...) list
        if sample.ndim == 5:
            items = [sample[i] for i in range(sample.shape[0])]
        else:
            items = [sample]

        records: List[Dict[str, Any]] = []
        for item in items:
            filename = self.filename_template.format(index=self._index)
            path = self.out_dir / filename

            arr = self._to_numpy_zyx(item)
            img = nib.Nifti1Image(arr, affine_arr)
            nib.save(img, str(path))

            rec = dict(meta)
            rec[self.image_key] = str(path)
            rec["index"] = self._index
            self._records.append(rec)
            records.append(rec)
            self._index += 1
        return records

    def flush(self) -> None:
        """Write current records to JSON file."""
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=self._json_indent)

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)
