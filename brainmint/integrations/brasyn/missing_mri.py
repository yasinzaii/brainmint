"""BraSyn / BrainLesion MissingMRI tensor translation wrappers."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from brainmint.integrations.brasyn.io import (
    _affine_from_tensor,
    _create_zero,
    _extract_filename_from_tensor,
    _nifti_to_tensor,
    _read_nifti,
    _tensor_to_nifti,
    center_crop_or_pad,
    ensure_b1hwd,
)
from brainmint.integrations.brasyn.modalities import GM_TO_BRAKEY
from brainmint.integrations.brasyn.runtime import (
    BraSynRuntime,
    _ensure_brats_installed,
    _patch_brats_singularity_runner,
    resolve_singularity_image_dir,
    resolve_tmp_root,
)


class BraSynMissingMRIGenerator:
    """Callable BrainLesion MissingMRI wrapper used by translation metrics/jobs."""

    def __init__(
        self,
        algorithm: str,
        backend: str = "singularity",
        cuda_devices: str = "0",
        runtime: dict[str, Any] | BraSynRuntime | None = None,
        reference_modality: str = "t1w",
        out_hwd: Sequence[int] | None = None,
        brats_hwd: Sequence[int] = (240, 240, 155),
    ) -> None:
        _ensure_brats_installed()
        from brats import MissingMRI
        from brats.constants import MissingMRIAlgorithms

        if isinstance(runtime, BraSynRuntime):
            self.runtime = runtime
        else:
            runtime = runtime or {}
            self.runtime = BraSynRuntime(
                tmp_root=str(resolve_tmp_root(runtime.get("tmp_root"))),
                overlay_mode=str(runtime.get("overlay_mode", "tmpfs")),
                overlay_size_mb=int(runtime.get("overlay_size_mb", 256)),
                overlay_readonly=bool(runtime.get("overlay_readonly", True)),
                keep_overlay=bool(runtime.get("keep_overlay", True)),
                use_fakeroot=bool(runtime.get("use_fakeroot", False)),
                singularity_image_dir=(
                    str(runtime.get("singularity_image_dir"))
                    if runtime.get("singularity_image_dir")
                    else None
                ),
            )

        self.runtime.tmp_root = str(resolve_tmp_root(self.runtime.tmp_root))
        resolved_image_dir = resolve_singularity_image_dir(self.runtime.singularity_image_dir)
        self.runtime.singularity_image_dir = str(resolved_image_dir) if resolved_image_dir else None

        self.algorithm_name = str(algorithm)
        self.backend = str(backend).lower()
        self.cuda_devices = str(cuda_devices)
        self.reference_modality = str(reference_modality).lower()
        self.brats_hwd = tuple(int(v) for v in brats_hwd)
        self.out_hwd = tuple(int(v) for v in out_hwd) if out_hwd is not None else None
        self._logger = logging.getLogger(__name__)

        if self.backend in {"singularity", "apptainer"}:
            _patch_brats_singularity_runner(self.runtime)

        enum = getattr(MissingMRIAlgorithms, self.algorithm_name)
        self._missing_mri = MissingMRI(algorithm=enum, cuda_devices=self.cuda_devices)

    def _emit_brasyn_log(self, log_path: str | Path, prefix: str = "") -> None:
        path = Path(log_path)
        if not path.exists():
            self._logger.warning("%sBraSyn log file missing: %s", prefix, path)
            return
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line:
                        self._logger.info("%s%s", prefix, line)
        except Exception:
            self._logger.exception("%sFailed reading BraSyn log: %s", prefix, path)

    def __call__(self, batch: Mapping[str, Any], modality: str) -> torch.Tensor:
        """Generate the requested modality from tensor conditioning inputs."""

        target = str(modality).lower()
        if target not in GM_TO_BRAKEY:
            raise KeyError(f"Unsupported target modality: {modality}")
        if target == "t1w":
            raise ValueError("BraSynMissingMRIGenerator does not support generating t1w; t1w must be provided.")

        t1w_tensor = batch.get("t1w")
        if not torch.is_tensor(t1w_tensor):
            raise ValueError("BraSynMissingMRIGenerator requires a t1w tensor in batch['t1w'].")

        ref_path = _extract_filename_from_tensor(t1w_tensor)
        affine = None
        if ref_path is not None:
            try:
                _, meta = _read_nifti(ref_path)
                affine = meta.get("affine")
            except Exception:
                affine = _affine_from_tensor(t1w_tensor)
        else:
            affine = _affine_from_tensor(t1w_tensor)

        with tempfile.TemporaryDirectory(
            prefix="brasyn_missingmri_",
            dir=str(self.runtime.tmp_root) if self.runtime.tmp_root else None,
        ) as tmp_dir:
            tmp_path = Path(tmp_dir)
            zero_path = tmp_path / "zero.nii.gz"
            _create_zero(self.brats_hwd, zero_path, affine=affine)

            cond_paths: dict[str, str] = {}
            for gm_mod, br_key in GM_TO_BRAKEY.items():
                if gm_mod == target:
                    continue
                tensor = batch.get(gm_mod)
                if torch.is_tensor(tensor):
                    out_path = tmp_path / f"in_{gm_mod}.nii.gz"
                    tensor = center_crop_or_pad(ensure_b1hwd(tensor), self.brats_hwd)
                    _tensor_to_nifti(tensor, out_path, affine=affine)
                    cond_paths[br_key] = str(out_path)
                else:
                    cond_paths[br_key] = str(zero_path)
                self._logger.info("BraSyn conditioning input: %s -> %s (%s)", gm_mod, br_key, cond_paths[br_key])

            out_tmp = tmp_path / f"out_{target}.nii.gz"
            log_file = tmp_path / "brats.log"
            self._missing_mri.infer_single(
                output_file=str(out_tmp),
                log_file=str(log_file),
                backend=self.backend,
                **cond_paths,
            )

            if log_file.exists():
                self._emit_brasyn_log(str(log_file), prefix=f"[BraSyn][{target}] ")

            out_tensor = _nifti_to_tensor(out_tmp)
            if self.out_hwd is not None:
                out_tensor = center_crop_or_pad(out_tensor, self.out_hwd)
            return out_tensor


class BraSyn2023Pix2PixGenerator(BraSynMissingMRIGenerator):
    """Ivo Baltruschat 2023 Pix2Pix MissingMRI baseline."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(algorithm="BraTS23_1", **kwargs)


class BraSyn2024HFGANRefinerGenerator(BraSynMissingMRIGenerator):
    """Jihoon Cho et al. 2024 HF-GAN + 3D Refiner MissingMRI baseline."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(algorithm="BraTS24_1", **kwargs)


class BraSyn2024MMCLACAGANGenerator(BraSynMissingMRIGenerator):
    """Minjoo Lim et al. 2024 MMCL / ACA-GAN MissingMRI baseline."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(algorithm="BraTS24_3", **kwargs)
