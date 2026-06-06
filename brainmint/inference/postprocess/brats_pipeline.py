from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_to_output

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import Postprocessor
from brainmint.utils.spatial import center_crop_or_pad_3d

log = logging.getLogger(__name__)


def _as_3tuple_int(x: Sequence[int], *, name: str) -> Tuple[int, int, int]:
    t = tuple(int(v) for v in x)
    if len(t) != 3:
        raise ValueError(f"{name} must have length 3, got {len(t)}")
    return t  # (Z,Y,X)


def _as_3tuple_float(x: Sequence[float], *, name: str) -> Tuple[float, float, float]:
    t = tuple(float(v) for v in x)
    if len(t) != 3:
        raise ValueError(f"{name} must have length 3, got {len(t)}")
    return t  # (sx, sy, sz) in NIfTI XYZ


def _require_brainles_preprocessing() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from brainles_preprocessing.brain_extraction import HDBetExtractor  # type: ignore
        from brainles_preprocessing.modality import CenterModality  # type: ignore
        from brainles_preprocessing.normalization.percentile_normalizer import PercentileNormalizer  # type: ignore
        from brainles_preprocessing.preprocessor import NativeSpacePreprocessor, Preprocessor  # type: ignore
        from brainles_preprocessing.registration import ANTsRegistrator  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "BratsPipelinePostprocess requires brainles_preprocessing for the BRATS-style "
            "preprocessing pipeline. Install brainles_preprocessing or configure a lighter postprocess."
        ) from exc

    return (
        ANTsRegistrator,
        CenterModality,
        HDBetExtractor,
        NativeSpacePreprocessor,
        PercentileNormalizer,
        Preprocessor,
    )


class BratsPipelinePostprocess(Postprocessor):
    """
    BrainScape BRATS-style postprocess for synthetic MRI.

    Per-sample:
      1) write decoded tensor to temp NIfTI (stored array in X,Y,Z) with input spacing
      2) run brainles_preprocessing BRATS-like pipeline:
           - optional atlas-centric registration (ANTs)
           - optional HD-BET
           - percentile normalization
      3) resample to target_spacing_xyz
      4) convert back to (Z,Y,X) and center crop/pad to roi_size_zyx
    """

    def __init__(
        self,
        *,
        roi_size_zyx: Sequence[int],
        input_spacing_xyz: Sequence[float] = (1.0, 1.0, 1.0),
        target_spacing_xyz: Sequence[float] = (1.0, 1.0, 1.0),
        skip_brain_extraction: bool = False,
        enable_registration: bool = True,
        limit_cuda_visible_devices: Optional[str] = "0",  # kept for hydra compat; don't set env here
        modality_key: str = "modality",
        resample_order: int = 1,
        temp_prefix: str = "brainmint_brats_",
    ) -> None:
        super().__init__()

        self.roi_size_zyx = _as_3tuple_int(roi_size_zyx, name="roi_size_zyx")
        self.input_spacing_xyz = _as_3tuple_float(input_spacing_xyz, name="input_spacing_xyz")
        self.target_spacing_xyz = _as_3tuple_float(target_spacing_xyz, name="target_spacing_xyz")

        self.skip_brain_extraction = bool(skip_brain_extraction)
        self.enable_registration = bool(enable_registration)

        self.limit_cuda_visible_devices = limit_cuda_visible_devices

        self.modality_key = str(modality_key)
        self.resample_order = int(resample_order)
        self.temp_prefix = str(temp_prefix)

        if any(v <= 0 for v in self.roi_size_zyx):
            raise ValueError(f"roi_size_zyx must be positive, got {self.roi_size_zyx}")
        if any(v <= 0 for v in self.input_spacing_xyz):
            raise ValueError(f"input_spacing_xyz must be positive, got {self.input_spacing_xyz}")
        if any(v <= 0 for v in self.target_spacing_xyz):
            raise ValueError(f"target_spacing_xyz must be positive, got {self.target_spacing_xyz}")
        if not (0 <= self.resample_order <= 5):
            raise ValueError("resample_order must be in [0..5]")

        (
            ants_registrator_cls,
            self._center_modality_cls,
            hdbet_extractor_cls,
            self._native_preprocessor_cls,
            percentile_normalizer_cls,
            self._preprocessor_cls,
        ) = _require_brainles_preprocessing()

        self._run_idx = 0
        self._normalizer = percentile_normalizer_cls(
            lower_percentile=0.1,
            upper_percentile=99.9,
            lower_limit=0,
            upper_limit=1,
        )
        self._brain_extractor = None if self.skip_brain_extraction else hdbet_extractor_cls()
        self._registrator = ants_registrator_cls() if self.enable_registration else None

    def _hydra_run_dir(self) -> Path:
        try:
            from hydra.core.hydra_config import HydraConfig  # type: ignore

            return Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            return Path.cwd()

    def _log_file(self, *, sample_idx: int) -> Path:
        run_dir = self._hydra_run_dir()
        postprep_log_dir = run_dir / "post_prep_logs"
        postprep_log_dir.mkdir(parents=True, exist_ok=True)
        return postprep_log_dir / f"brats_preprocess_{self._run_idx:06d}_{sample_idx:04d}.log"

    @staticmethod
    def _tensor_zyx_to_nifti_xyz(vol_zyx: torch.Tensor, *, affine_xyz: np.ndarray, out_path: Path) -> None:
        arr_zyx = np.asarray(vol_zyx.detach().cpu(), dtype=np.float32)
        if arr_zyx.ndim != 3:
            raise ValueError(f"Expected 3D (Z,Y,X) volume, got shape={arr_zyx.shape}")
        arr_xyz = np.transpose(arr_zyx, (2, 1, 0))  # (Z,Y,X) -> (X,Y,Z)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(arr_xyz, affine=affine_xyz.astype(np.float32)), str(out_path))

    @staticmethod
    def _nifti_xyz_to_numpy_zyx(img: Any) -> np.ndarray:
        arr_xyz = np.squeeze(img.get_fdata(dtype=np.float32))
        if arr_xyz.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI after preprocessing, got shape={arr_xyz.shape}")
        return np.transpose(arr_xyz, (2, 1, 0))  # (X,Y,Z) -> (Z,Y,X)

    def _center_crop_or_pad(self, arr_zyx: np.ndarray) -> np.ndarray:
        if arr_zyx.ndim != 3:
            raise ValueError(f"Expected 3D array (Z,Y,X), got shape={arr_zyx.shape}")
        out = np.asarray(center_crop_or_pad_3d(arr_zyx, self.roi_size_zyx), dtype=np.float32)
        if out.shape != tuple(self.roi_size_zyx):
            raise ValueError(f"Crop/pad produced shape {out.shape}, expected {self.roi_size_zyx}")
        return out

    def _resolve_modalities_from_metadata(self, *, batch_size: int, metadata: Mapping[str, Any]) -> List[str]:
        if self.modality_key not in metadata:
            raise KeyError(
                f"Missing metadata['{self.modality_key}'] for BRATS postprocess. "
                f"Provide per-sample modality names (list/tuple length B={batch_size})."
            )

        raw = metadata[self.modality_key]

        if isinstance(raw, str):
            if batch_size != 1:
                raise ValueError(
                    f"metadata['{self.modality_key}'] is a single string ('{raw}') but batch_size={batch_size}. "
                    "Provide a list/tuple of modality names with length == batch_size."
                )
            return [raw.strip().lower()]

        if isinstance(raw, (list, tuple)):
            if len(raw) != batch_size:
                raise ValueError(f"metadata['{self.modality_key}'] length {len(raw)} != batch_size {batch_size}")
            return [str(v).strip().lower() for v in raw]

        if isinstance(raw, np.ndarray):
            flat = raw.reshape(-1).tolist()
            if len(flat) != batch_size:
                raise ValueError(f"metadata['{self.modality_key}'] length {len(flat)} != batch_size {batch_size}")
            return [str(v).strip().lower() for v in flat]

        if torch.is_tensor(raw):
            flat = raw.detach().cpu().reshape(-1).tolist()
            if len(flat) != batch_size:
                raise ValueError(f"metadata['{self.modality_key}'] length {len(flat)} != batch_size {batch_size}")
            return [str(v).strip().lower() for v in flat]

        raise TypeError(
            f"Unsupported type for metadata['{self.modality_key}']: {type(raw)}. "
            "Use list/tuple[str] length B (or a string only if B==1)."
        )

    def _run_brats_preprocessor(
        self,
        *,
        input_path: Path,
        work_dir: Path,
        modality_name: str,
        log_file: Path,
    ) -> Path:
        norm_bet = work_dir / "norm_bet"
        temp_dir = work_dir / "temp"
        norm_bet.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        stem = input_path.name[:-7] if input_path.name.endswith(".nii.gz") else Path(input_path.name).stem

        if self.skip_brain_extraction:
            out_path = norm_bet / f"{stem}_raw_skull_{modality_name}.nii.gz"
        else:
            out_path = norm_bet / f"{stem}_norm_bet_{modality_name}.nii.gz"

        atlas_correction = bool(self.enable_registration)

        if self.skip_brain_extraction:
            center = self._center_modality_cls(
                modality_name=modality_name,
                input_path=input_path,
                normalized_skull_output_path=out_path,
                atlas_correction=atlas_correction,
                normalizer=self._normalizer,
            )
        else:
            center = self._center_modality_cls(
                modality_name=modality_name,
                input_path=input_path,
                normalized_bet_output_path=out_path,
                atlas_correction=atlas_correction,
                normalizer=self._normalizer,
            )

        if self.enable_registration:
            pre = self._preprocessor_cls(
                center_modality=center,
                moving_modalities=[],
                registrator=self._registrator,
                brain_extractor=self._brain_extractor,
                temp_folder=temp_dir,
            )
        else:
            pre = self._native_preprocessor_cls(
                center_modality=center,
                moving_modalities=[],
                registrator=None,
                brain_extractor=self._brain_extractor,
                temp_folder=temp_dir,
            )

        pre.run(log_file=log_file)

        if not out_path.exists():
            raise FileNotFoundError(f"Expected BRATS preprocessed output not found: {out_path}")

        return out_path

    def process(self, x: torch.Tensor, *, ctx: Optional[InferenceContext] = None) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B,C,Z,Y,X), got shape={tuple(x.shape)}")
        if ctx is None:
            raise ValueError("BratsPipelinePostprocess requires ctx with ctx.metadata (for modality list).")

        b, c = int(x.shape[0]), int(x.shape[1])
        if c != 1:
            raise ValueError(f"BratsPipelinePostprocess supports C=1 only, got C={c}")

        modality_names = self._resolve_modalities_from_metadata(batch_size=b, metadata=ctx.metadata)

        sx, sy, sz = self.input_spacing_xyz
        affine_xyz = np.eye(4, dtype=np.float32)
        affine_xyz[0, 0] = sx
        affine_xyz[1, 1] = sy
        affine_xyz[2, 2] = sz

        out = torch.empty((b, 1, *self.roi_size_zyx), device=x.device, dtype=x.dtype)

        for i in range(b):
            log_file = self._log_file(sample_idx=i)

            with TemporaryDirectory(prefix=self.temp_prefix) as td:
                work_dir = Path(td)
                input_path = work_dir / "synthetic.nii.gz"

                self._tensor_zyx_to_nifti_xyz(x[i, 0], affine_xyz=affine_xyz, out_path=input_path)

                pre_out_path = self._run_brats_preprocessor(
                    input_path=input_path,
                    work_dir=work_dir,
                    modality_name=modality_names[i],
                    log_file=log_file,
                )

                img = nib.load(str(pre_out_path))

                img_rs = resample_to_output(
                    img,
                    voxel_sizes=self.target_spacing_xyz,
                    order=self.resample_order,
                )

                arr_zyx = self._nifti_xyz_to_numpy_zyx(img_rs)
                arr_zyx = self._center_crop_or_pad(arr_zyx)

                out[i, 0] = torch.from_numpy(arr_zyx).to(device=x.device, dtype=x.dtype)

        self._run_idx += 1
        return out
