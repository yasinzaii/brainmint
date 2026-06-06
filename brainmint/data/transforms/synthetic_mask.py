"""Synthetic mask generation transform for BrainScape MRI data.

This module augments an input dictionary with tumour segmentation masks for
inpainting tasks. When a sample already contains a segmentation mask on disk
it will be loaded and converted into multi-channel Boolean masks. When the
sample does not have a mask, a random mask is drawn from the BraTS training
set, spatially warped and matched to the current image size. Augmentations
include random axis flips, rotations and affine warps. A dilated version of
the full tumour mask is also computed to cover tissue surrounding the lesion.
The transform additionally applies this dilated mask to the original image to
produce a masked image that can be passed to ControlNet for context.

Returned keys::

    mask_one_hot (Tensor): Boolean tensor of shape ``(C, D, H, W)`` with
        channels ``[HEALTHY, NCR_NET, ED, ET]``.
    mask_is_synthetic (bool): ``True`` if the mask was generated from the
        pool rather than loaded from disk.
    mask_dilated (Tensor): Boolean tensor of shape ``(1, D, H, W)``
        representing the dilated full tumour region.
    image_masked (Tensor): Tensor of shape ``(C_img, D, H, W)`` where the
        dilated tumour region has been zeroed out.

The transform expects the input dict to contain at least ``image`` and,
optionally, ``seg``. If ``seg`` is absent, a synthetic mask is sampled from
the BraTS mask pool.

Parameters such as the dilation range and augmentation probabilities are
configurable via the constructor. See the class docstring for details.

Note: This transform caches loaded NIfTI masks in memory as 8-bit arrays to
avoid repeated disk I/O. Set ``cache_masks=False`` to disable in-memory
caching if memory becomes a constraint.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import scipy.ndimage as ndi
except ImportError as exc:  # pragma: no cover - exercised only without optional extra
    raise ImportError(
        "scipy is required for SyntheticMaskTransform. Install it with `pip install brainmint[medical]`."
    ) from exc
import torch
import torch.nn.functional as F
from monai.transforms import Affine, MapTransform
from monai.transforms.transform import Randomizable

log = logging.getLogger(__name__)


class SyntheticMaskTransform(Randomizable, MapTransform):
    """
    Generate or load tumour segmentation masks and augment the sample.

    This transform operates on dictionary entries to either load an
    existing segmentation mask or synthesise a mask from a pool of real
    BraTS masks when no mask is provided.  It converts the mask into
    one-hot channels, computes a dilated region around the tumour and
    applies the dilated mask to the image for inpainting.  The resulting
    tensors are inserted into the dict with new keys.

    Args:
        image_key: Key whose value is the image (tensor).
        mask_key: Optional key for an existing segmentation mask.  When
            missing or ``None`` a synthetic mask will be generated.
        mask_loader_tf: Transform used to load a segmentation mask
            from a file path when ``mask_key`` contains a string.
        dataset_json: Path to the BrainScape dataset JSON file.  BraTS
            segmentation masks will be collected from ``train`` set.
        dataset_root: Root directory of the preprocessed BrainScape data.
            Mask paths from the JSON are resolved relative to this root.
        dilation_mm_range: Tuple specifying the minimum and maximum
            dilation in millimetres applied to the full tumour mask to
            produce ``mask_dilated``.  The BrainScape dataset has
            isotropic 1 mm spacing, so millimetres equal voxels.
        prob_flip: Probability of flipping the synthetic mask along
            each spatial axis.  Should be in [0, 1].
        prob_rotate: Probability of randomly rotating the synthetic mask
            by up to ``max_rotate`` degrees about each axis.
        prob_affine: Probability of applying a random affine warp
            including scaling, shearing, rotation and translation.
        max_rotate: Maximum absolute rotation (degrees) applied during
            random rotations and affine transforms.
        max_scale: Maximum fractional scaling applied in random affine
            transforms. Scaling is applied symmetrically for up- and
            down-sampling.
        max_shear: Maximum shear magnitude for random affine transforms.
        max_translate: Maximum translation (voxels) for random affine
            transforms.
        seed: Optional integer seed for deterministic behaviour in tests.
        cache_masks: Whether to cache loaded masks in memory as 8-bit arrays
            to avoid repeated disk reads.
    """

    def __init__(
        self,
        image_key: str,
        mask_key: Optional[str],
        mask_loader_tf,  # Transform Compose
        dataset_json: Union[str, Path],
        dataset_root: Union[str, Path],
        dilation_mm_range: Tuple[int, int] = (5, 10),
        prob_flip: float = 0.5,
        prob_rotate: float = 0.5,
        prob_affine: float = 0.7,
        max_rotate: float = 15.0,
        max_scale: float = 0.15,
        max_shear: float = 0.15,
        max_translate: float = 5.0,
        seed: Optional[int] = 42,
        cache_masks: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        keys = [image_key] + ([mask_key] if mask_key is not None else [])
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.image_key = image_key
        self.mask_key = mask_key
        self.mask_loader_tf = mask_loader_tf
        self.dataset_json = Path(dataset_json)
        self.dataset_root = Path(dataset_root)
        self.mask_pool: List[str] = self._gather_mask_pool()
        if not self.mask_pool:
            raise ValueError("No BraTS segmentation masks found in dataset JSON")
        self.dilation_mm_range = (max(0, int(dilation_mm_range[0])), max(0, int(dilation_mm_range[1])))
        self.prob_flip = float(prob_flip)
        self.prob_rotate = float(prob_rotate)
        self.prob_affine = float(prob_affine)
        self.max_rotate = float(max_rotate)
        self.max_scale = float(max_scale)
        self.max_shear = float(max_shear)
        self.max_translate = float(max_translate)
        self.rng = np.random.default_rng(seed)
        self.cache_masks = bool(cache_masks)
        self._mask_cache: Dict[str, np.ndarray] = {}

    def set_random_state(self, seed=None, state=None):
        super().set_random_state(seed, state)
        self.rng = np.random.default_rng(seed)

    def _load_mask(self, path: str) -> np.ndarray:
        """Load a NIfTI segmentation mask and return a numpy array of ints.

        If ``cache_masks`` is enabled, loaded masks are stored in memory as
        ``uint8`` arrays and subsequent requests for the same path will return
        a copy from the cache.
        """
        if self.cache_masks and path in self._mask_cache:
            return self._mask_cache[path].copy()

        data = self.mask_loader_tf({self.mask_key: path})
        arr = data.get(self.mask_key)
        if isinstance(arr, torch.Tensor):
            arr = arr.squeeze().cpu().numpy()
        else:
            arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0]

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D mask after load; got {arr.shape} from {path}")

        if self.cache_masks:
            self._mask_cache[path] = arr
            return arr.copy()
        return arr

    def _gather_mask_pool(self) -> List[str]:
        """Collect BRATS segmentation mask paths from the dataset JSON."""
        with self.dataset_json.open("r") as f:
            meta = json.load(f)
        masks: List[str] = []
        train = meta.get("train")
        records = [rec for rec in train if rec.get("dataset") == "BRATS"]
        for rec in records:
            seg_path = rec.get("preprocessed").get("seg")
            if isinstance(seg_path, str):
                masks.append(str(self.dataset_root / "BRATS" / "preprocessed" / seg_path))
            else:
                raise ValueError(f"Segmentation (SEG) Label must be string, got: {seg_path}")
        return masks

    def _random_mask_from_pool(self) -> np.ndarray:
        """Pick a random mask from the pool and load it."""
        idx = self.rng.integers(len(self.mask_pool))
        mask_path = self.mask_pool[int(idx)]
        return self._load_mask(mask_path)

    def _random_transform(self, mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Apply random spatial augmentations and resize to ``target_shape``.

        Augmentations include axis flips, small-angle rotations and optional
        affine warps (scaling, shear, rotation, translation).  Nearest-neighbour
        interpolation is used throughout to preserve integer labels.
        """
        m = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        # Flip axes independently
        for axis in range(3):
            if self.rng.random() < self.prob_flip:
                m = torch.flip(m, dims=[axis + 1])
        # Rotate around each axis by a small random angle using MONAI's Affine
        angles = [
            float(self.rng.uniform(-self.max_rotate, self.max_rotate)) if self.rng.random() < self.prob_rotate else 0.0
            for _ in range(3)
        ]

        if any(a != 0.0 for a in angles):
            rot_tf = Affine(
                rotate_params=tuple(np.deg2rad(np.asarray(angles, dtype=np.float32))),
                mode="nearest",
                padding_mode="border",
                spatial_size=target_shape,
                image_only=True,
            )
            m = rot_tf(m)

        if self.rng.random() < self.prob_affine:
            rot = np.deg2rad(self.rng.uniform(-self.max_rotate, self.max_rotate, size=3).astype(np.float32))
            scale = (1.0 + self.rng.uniform(-self.max_scale, self.max_scale, size=3)).astype(np.float32)
            shear = self.rng.uniform(-self.max_shear, self.max_shear, size=6).astype(np.float32)
            translate = self.rng.uniform(-self.max_translate, self.max_translate, size=3).astype(np.float32)

            aff_tf = Affine(
                rotate_params=tuple(float(x) for x in rot),
                scale_params=tuple(float(x) for x in scale),
                shear_params=tuple(float(x) for x in shear),
                translate_params=tuple(float(x) for x in translate),
                mode="nearest",
                padding_mode="border",
                spatial_size=target_shape,
                image_only=True,
            )
            m = aff_tf(m)

        # Resize using nearest neighbour if needed
        if list(m.shape[1:]) != list(target_shape):
            m = F.interpolate(m.unsqueeze(0), size=target_shape, mode="nearest").squeeze(0)
        return m.squeeze(0).byte().cpu().numpy()

    def _to_one_hot(self, mask: np.ndarray) -> np.ndarray:
        """Convert single-channel label map into one-hot channels [H,NCR_NET,ED,ET]."""
        # BraTS labels: 0=background/healthy, 1=NCR/NET, 2=ED, 4=ET
        H = mask == 0
        ncr = mask == 1
        ed = mask == 2
        et = mask == 3
        # stack as boolean channels (C,D,H,W)
        return np.stack([H, ncr, ed, et], axis=0)

    def _dilate_mask(self, full_mask: np.ndarray, mm: int) -> np.ndarray:
        if mm <= 0:
            return full_mask.astype(bool, copy=True)
        r = int(mm)
        zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
        ball = (xx * xx + yy * yy + zz * zz) <= r * r
        return ndi.binary_dilation(full_mask.astype(bool), structure=ball)

    def __call__(self, data: Dict[str, Union[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        d = dict(data)
        # Load the image tensor (expects shape (C_img, D, H, W))
        img = d[self.image_key]
        img_tensor = img if isinstance(img, torch.Tensor) else torch.as_tensor(img)
        if img_tensor.ndim != 4:
            raise ValueError(f"Expected image as (C, D, H, W); got {tuple(img_tensor.shape)}")
        _, D, H, W = img_tensor.shape
        target_shape = (D, H, W)

        # Load or synthesise mask
        mask_arr = None
        if self.mask_key and self.mask_key in d and d[self.mask_key] is not None:
            mask_val = d[self.mask_key]
            if isinstance(mask_val, torch.Tensor):
                mask_arr = mask_val.detach().cpu().squeeze().to(torch.uint8).numpy()
            else:
                raise ValueError(f"Unsupported mask type for '{self.mask_key}': {type(mask_val)}")
        if mask_arr is None:
            # Synthesise
            mask_arr = self._random_mask_from_pool()
            mask_arr = self._random_transform(mask_arr, target_shape)
            is_synthetic = True
        else:
            # Ensure it has same shape
            if mask_arr.shape != target_shape:
                raise ValueError(f"Mask Shape and Image Shape Mismatch, Mask Shape:{mask_arr.shape}, Image Shape:{target_shape}")
            is_synthetic = False
        # Convert to one hot
        one_hot = self._to_one_hot(mask_arr)  # shape (4,D,H,W)
        # Compute full tumour mask (any class >0)
        full_mask = (mask_arr > 0)
        # Choose random dilation radius in mm
        dil_min, dil_max = self.dilation_mm_range
        if dil_max > dil_min:
            rad = int(self.rng.integers(dil_min, dil_max + 1))
        else:
            rad = int(dil_min)
        dilated = self._dilate_mask(full_mask, rad)
        # Apply mask to image: zero out dilated tumour region for context
        masked_img = img_tensor.clone()
        dil_tensor = torch.from_numpy(dilated).to(dtype=masked_img.dtype, device=masked_img.device).unsqueeze(0)
        masked_img = masked_img * (1.0 - dil_tensor)
        # Update dict
        d["mask_one_hot"] = torch.from_numpy(one_hot).to(device=img_tensor.device)
        d["mask_is_synthetic"] = bool(is_synthetic)
        d["mask_dilated"] = torch.from_numpy(dilated[None, ...]).to(device=img_tensor.device)
        d["image_masked"] = masked_img

        # ``seg`` is only an intermediate key required to load existing masks.
        # Drop it from the output dictionary so batches remain consistent when
        # some samples do not provide segmentation maps (e.g. non-BraTS data).
        if self.mask_key is not None:
            d.pop(self.mask_key, None)

        return d


class CachedMaskPoolOneHotTransform(Randomizable, MapTransform):
    """Return a random segmentation mask from a cached pool as a 4-channel one-hot.

    This is a **minimal** variant intended for Med-DDPM sampling. It:
      - builds a pool of BraTS segmentation masks from the BrainScape JSON (configurable split),
      - optionally caches them in memory,
      - uses an existing segmentation mask in the sample when provided,
      - on each call, selects a random mask and converts it to one-hot tumour channels,
      - derives the *brain area* channel deterministically from the current image.

    No geometric augmentations (flip/rotate/affine/dilation) are applied.

    Output key:
      - ``out_key``: float tensor of shape (4, D, W, H) with channels
        [TUMORAREA1, TUMORAREA2, TUMORAREA3, BRAINAREA].

    Notes:
      - BrainScape BraTS seg labels are expected to be {0,1,2,3} for tumour
        regions. Brain area is derived from the current image by selecting
        non-zero voxels and removing tumour regions.
    """

    def __init__(
        self,
        keys,
        *,
        image_key: str,
        out_key: str = "med_ddpm_mask_one_hot",
        mask_key: Optional[str] = "seg",
        mask_loader_tf,
        dataset_json: Union[str, Path],
        dataset_root: Union[str, Path],
        # where to gather seg masks from in the BrainScape JSON
        split: str = "train",
        group_key: str = "preprocessed",
        dataset_name_contains: Optional[str] = None,
        seed: Optional[int] = 42,
        cache_masks: bool = True,
        max_masks: Optional[int] = None,
        allow_missing_keys: bool = False,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.image_key = str(image_key)
        self.out_key = str(out_key)
        self.mask_key = str(mask_key) if mask_key is not None else None

        self.mask_loader_tf = mask_loader_tf
        self.dataset_json = Path(dataset_json)
        self.dataset_root = Path(dataset_root)

        self.split = str(split)
        self.group_key = str(group_key)
        self.dataset_name_contains = dataset_name_contains

        self.cache_masks = bool(cache_masks)
        self.max_masks = int(max_masks) if max_masks is not None else None
        self.output_dtype = output_dtype

        # Randomizable API
        self.set_random_state(seed=seed)
        self._base_seed = seed

        self._mask_paths: List[Path] = self._gather_mask_paths(
            dataset_json=self.dataset_json,
            dataset_root=self.dataset_root,
            split=self.split,
            group_key=self.group_key,
            dataset_name_contains=self.dataset_name_contains,
        )
        if self.max_masks is not None:
            self._mask_paths = self._mask_paths[: self.max_masks]
        if not self._mask_paths:
            raise RuntimeError(
                f"No seg masks found in JSON '{self.dataset_json}' for split='{self.split}' "
                f"(group_key='{self.group_key}', dataset_name_contains={self.dataset_name_contains!r})."
            )

        self._cache: Dict[str, np.ndarray] = {}
        if self.cache_masks:
            for p in self._mask_paths:
                _ = self._load_mask_np(p)

    @staticmethod
    def _gather_mask_paths(
        dataset_json: Path,
        dataset_root: Path,
        *,
        split: str = "train",
        group_key: str = "preprocessed",
        dataset_name_contains: Optional[str] = None,
    ) -> List[Path]:
        """Collect segmentation mask paths from BrainScape JSON.

        Args:
            dataset_json: BrainScape JSON file.
            dataset_root: Root folder containing <dataset_id>/preprocessed/<relpath>.
            split: Which split to pull masks from ("train"/"val"/"test").
            group_key: Which group dict to read from each record (default "preprocessed").
            dataset_name_contains: If set, only keep records whose "dataset" contains this
                substring (case-insensitive). Useful to restrict to BraTS.
        """
        source = json.loads(dataset_json.read_text())
        if split not in source:
            raise KeyError(f"Split '{split}' missing from dataset_json: {dataset_json}")
        out: List[Path] = []
        needle = dataset_name_contains.lower() if isinstance(dataset_name_contains, str) and dataset_name_contains else None

        for rec in source[split]:
            ds = rec.get("dataset")
            if not isinstance(ds, str) or not ds:
                continue
            if needle is not None and needle not in ds.lower():
                continue

            group = rec.get(group_key)
            if not isinstance(group, dict):
                continue

            for mod, rel in group.items():
                if not isinstance(mod, str):
                    continue
                if mod.lower() not in ("seg", "segmentation", "mask"):
                    continue
                if rel is None:
                    continue
                out.append((dataset_root / ds / str(group_key) / str(rel)).resolve())
        return out

    def _load_mask_np(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._cache:
            return self._cache[key]

        # Use the provided loader transform to apply the same deterministic
        # crop/pad as the main image pipeline.
        loaded = self.mask_loader_tf({"seg": str(path)})
        seg = loaded.get("seg")
        if seg is None:
            raise RuntimeError(f"mask_loader_tf did not return 'seg' for: {path}")
        if isinstance(seg, torch.Tensor):
            seg_np = seg.detach().cpu().numpy()
        else:
            seg_np = np.asarray(seg)

        # Ensure (D,H,W) - VAETransform typically yields (1,D,H,W)
        if seg_np.ndim == 4 and seg_np.shape[0] == 1:
            seg_np = seg_np[0]
        if seg_np.ndim != 3:
            raise ValueError(f"Expected seg (D,H,W), got {seg_np.shape} for {path}")

        seg_np = seg_np.astype(np.uint8, copy=False)
        if self.cache_masks:
            self._cache[key] = seg_np
        return seg_np

    @staticmethod
    def _to_tumour_one_hot(seg: np.ndarray) -> np.ndarray:
        TUMORAREA1 = 1
        TUMORAREA2 = 2
        TUMORAREA3 = 3

        ch0 = (seg == TUMORAREA1)
        ch1 = (seg == TUMORAREA2)
        ch2 = (seg == TUMORAREA3)

        return np.stack([ch0, ch1, ch2], axis=0).astype(np.uint8)

    @staticmethod
    def _brain_from_image(img: torch.Tensor) -> torch.Tensor:
        """Compute brain area mask from the current image tensor.

        Expects img shape (C,D,H,W) or (D,H,W). Returns (D,H,W) bool.
        """
        if img.dim() == 4:
            # Any non-zero across channels
            return img.abs().sum(dim=0) > 0
        if img.dim() == 3:
            return img.abs() > 0
        raise ValueError(f"Unexpected image dim {img.dim()} for brain mask")

    def _get_seg_from_sample(self, sample: Dict) -> Optional[np.ndarray]:
        if not self.mask_key:
            return None
        if self.mask_key not in sample:
            return None
        mask_val = sample.get(self.mask_key)
        if mask_val is None:
            return None
        if isinstance(mask_val, (str, Path)):
            return self._load_mask_np(Path(mask_val))
        if isinstance(mask_val, torch.Tensor):
            seg_np = mask_val.detach().cpu().numpy()
        else:
            seg_np = np.asarray(mask_val)
        if seg_np.ndim == 4 and seg_np.shape[0] == 1:
            seg_np = seg_np[0]
        if seg_np.ndim != 3:
            raise ValueError(f"Expected seg (D,H,W), got {seg_np.shape} for key '{self.mask_key}'")
        return seg_np.astype(np.uint8, copy=False)

    @staticmethod
    def _center_crop_np(seg_np: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Center-crop seg_np (D,H,W) to target_shape."""
        d, h, w = seg_np.shape
        td, th, tw = target_shape
        sd = (d - td) // 2
        sh = (h - th) // 2
        sw = (w - tw) // 2
        return seg_np[sd:sd + td, sh:sh + th, sw:sw + tw]

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)

        if self.image_key not in d:
            raise KeyError(f"CachedMaskPoolOneHotTransform expects key '{self.image_key}'")

        img = d[self.image_key]
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)

        seg_np = self._get_seg_from_sample(d)
        seg_path: Optional[Path] = None
        if seg_np is None:
            # Pick random BraTS seg label map from the cached pool.
            path = self._mask_paths[int(self.R.randint(0, len(self._mask_paths)))]
            seg_np = self._load_mask_np(path)  # (D,H,W) uint8
            seg_path = path

        image_shape = tuple(int(x) for x in img.shape[-3:])
        seg_shape = tuple(int(x) for x in seg_np.shape[-3:])
        if image_shape != seg_shape:
            if seg_path is not None:
                meta_strings = {k: v for k, v in d.items() if isinstance(v, str)}
                log.warning(
                    "CachedMaskPoolOneHotTransform shape mismatch for pooled seg path=%s | image_shape=%s seg_shape=%s | string_meta=%s",
                    str(seg_path),
                    image_shape,
                    seg_shape,
                    meta_strings,
                )
            img_smaller_than_segmask = all(img_s < seg_s for img_s, seg_s in zip(image_shape, seg_shape))
            if img_smaller_than_segmask:
                seg_np = self._center_crop_np(seg_np, image_shape)
            else:
                raise RuntimeError(
                    "CachedMaskPoolOneHotTransform received unexpected shape mismatch: "
                    f"image_shape={image_shape}, seg_shape={seg_shape}. "
                    "Only center-crop of larger seg masks is supported."
                )

        tumour_oh = self._to_tumour_one_hot(seg_np)  # (3,D,H,W)
        tumour = torch.from_numpy(tumour_oh).to(device=img.device, dtype=torch.bool)

        brain = self._brain_from_image(img).to(device=img.device, dtype=torch.bool)
        brain = brain & ~(tumour.any(dim=0))

        mask = torch.cat([tumour, brain.unsqueeze(0)], dim=0).to(dtype=self.output_dtype)
        d[self.out_key] = mask
        return d
