from __future__ import annotations

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def make_minimal_brainscape_dataset(
    tmp_dataset_path: Path,
    blueprint_json: str | Path,
    *,
    image_size: int | tuple[int, int, int] | None = None,
) -> Path:
    """Create a tiny BrainScape-like dataset from a minimal BrainScape JSON file.

    Files are created as:

        <tmp_dataset_path>/<dataset>/preprocessed/<relative_path_from_json>

    If image_size is None, empty files are created with Path.touch().
    If image_size is an int or 3-tuple, small NIfTI files are written.

    Returns:
        Path to the copied JSON blueprint inside tmp_dataset_path.
    """

    tmp_dataset_path = Path(tmp_dataset_path)
    blueprint_json = Path(blueprint_json)

    tmp_dataset_path.mkdir(parents=True, exist_ok=True)
    json_path = tmp_dataset_path / blueprint_json.name
    shutil.copyfile(blueprint_json, json_path)

    data = json.loads(blueprint_json.read_text(encoding="utf-8"))
    shape = _resolve_shape(image_size)

    for split_name, records in data.items():
        if not isinstance(records, list):
            raise ValueError(f"Split {split_name!r} must contain a list of records")

        for record in records:
            dataset = str(record["dataset"])
            preprocessed = record.get("preprocessed", {})
            if not isinstance(preprocessed, dict):
                raise ValueError(f"Record {record.get('subject')!r} has invalid preprocessed field")

            for modality, rel_path in preprocessed.items():
                out_path = tmp_dataset_path / dataset / "preprocessed" / str(rel_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if shape is None:
                    out_path.touch()
                else:
                    _write_dummy_nifti(
                        out_path,
                        shape=shape,
                        is_label=str(modality).lower() == "seg",
                    )

    return json_path


def _resolve_shape(image_size: int | tuple[int, int, int] | None) -> tuple[int, int, int] | None:
    if image_size is None:
        return None
    if isinstance(image_size, int):
        return (image_size, image_size, image_size)

    shape = tuple(int(v) for v in image_size)
    if len(shape) != 3:
        raise ValueError(f"image_size must be an int or 3-tuple, got {image_size!r}")
    return shape


def _write_dummy_nifti(path: Path, *, shape: tuple[int, int, int], is_label: bool) -> None:
    if is_label:
        data = np.zeros(shape, dtype=np.uint8)
    else:
        data = np.random.default_rng(0).random(shape, dtype=np.float32)

    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, affine), str(path))