"""Segmentation-specific transforms.

These transforms are designed to be used via the existing VAE data pipeline
(`brainmint/data/transforms/mri_vae.py`) using `extra_xforms_start` / `extra_xforms_end`.

Primary use-case: train a VAE on **binary region masks** derived from a BraTS-style
label map (0=background, 1=NCR, 2=ED, 3=ET).

Key ideas:
  • Keep the label map intact (no intensity scaling).
  • Derive configurable region masks (NCR/ED/ET + unions like WT/TC).
  • Randomly select one region mask and copy it to `batch["image"]` so the existing
    `brainmint/lightning/vae_module.py` can be reused without modification.
  • Optionally compute a brain mask from a reference modality (threshold + optional erosion)
    and optionally invert masks **within brain** to increase robustness.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from monai.transforms import Transform



def _as_float_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.float()
    return torch.as_tensor(x).float()


def _binary_dilation(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation using max_pool3d.

    Expects `mask` shape (C, *spatial) or (B, C, *spatial).
    Returns float tensor with values in {0,1}.
    """
    if radius <= 0:
        return (mask > 0.5).float()

    if mask.dim() == 4:
        x = mask.unsqueeze(0)  # (1, C, X, Y, Z)
        squeeze_b = True
    elif mask.dim() == 5:
        x = mask
        squeeze_b = False
    else:
        raise ValueError(f"Expected 4D or 5D tensor for dilation, got shape {tuple(mask.shape)}")

    k = 2 * int(radius) + 1
    x = (x > 0.5).float()
    y = F.max_pool3d(x, kernel_size=k, stride=1, padding=radius)

    if squeeze_b:
        y = y.squeeze(0)
    return (y > 0.5).float()


class MakeBrainMaskd(Transform):
    """Compute a binary brain mask from a reference modality.

    Args:
        ref_key: Key containing the reference modality tensor.
        out_key: Key to store the resulting brain mask.
        eps: Threshold. Brain mask = ref > eps.
        drop_ref: If True, remove `ref_key` from the dictionary after computing the mask.
        allow_missing_keys: If True, silently skip when `ref_key` is absent.
    """

    def __init__(
        self,
        *,
        ref_key: str,
        out_key: str = "brain_mask",
        eps: float = 1e-4,
        drop_ref: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        self.ref_key = str(ref_key)
        self.out_key = str(out_key)
        self.eps = float(eps)
        self.drop_ref = bool(drop_ref)
        self.allow_missing_keys = bool(allow_missing_keys)

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.ref_key not in d:
            if self.allow_missing_keys:
                return d
            raise KeyError(f"MakeBrainMaskd: missing ref_key '{self.ref_key}'. Keys={list(d.keys())}")

        ref = _as_float_tensor(d[self.ref_key])
        mask = (ref > self.eps).float()

        d[self.out_key] = mask
        if self.drop_ref:
            d.pop(self.ref_key, None)
        return d


class BraTSLabelToRegionsd(Transform):
    """Convert a BraTS-style label map into configurable binary region masks.

    Expected label values (default):
        - background = 0
        - NCR = 1
        - ED  = 2
        - ET  = 3

    You can request any subset of output masks, including unions like WT and TC.

    Args:
        seg_key: Key containing the label map.
        regions_out: List of region names to emit as keys.
        label_values: Mapping from base region name -> integer label value.
        unions: Mapping from derived region name -> list of base region names.
        dtype: Output dtype (float masks recommended).
        drop_seg: If True, remove `seg_key` after conversion.
        allow_missing_keys: If True, skip when seg_key missing.
    """
    DEFAULT_LABEL_VALUES: Dict[str, int] = {"NCR": 1, "ED": 2, "ET": 3}
    DEFAULT_UNIONS: Dict[str, Sequence[str]] = {
        "WT": ("NCR", "ED", "ET"),
        "TC": ("NCR", "ET"),
    }
    
    def __init__(
        self,
        *,
        seg_key: str = "seg",
        regions_out: Sequence[str] = ("NCR", "ED", "ET"),
        label_values: Optional[Mapping[str, int]] = None,
        unions: Optional[Mapping[str, Sequence[str]]] = None,
        dtype: str | torch.dtype = torch.float32,
        drop_seg: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        self.seg_key = str(seg_key)
        self.regions_out = [str(r) for r in regions_out]

        if isinstance(dtype, str):
            s = dtype.strip().lower().removeprefix("torch.")
            if s in ("float", "float32", "fp32"):
                dtype = torch.float32
            else: 
               raise ValueError(f"BraTSLabelToRegionsd: unsupported dtype string '{dtype}'.") 
        self.dtype = dtype
        
        self.drop_seg = bool(drop_seg)
        self.allow_missing_keys = bool(allow_missing_keys)

        base = self.DEFAULT_LABEL_VALUES
        if label_values is not None:
            base = {str(k).upper(): int(v) for k, v in dict(label_values).items()}
        self.label_values = base

        default_unions = self.DEFAULT_UNIONS
        if unions is not None:
            default_unions = {str(k).upper(): tuple(str(x).upper() for x in v) for k, v in dict(unions).items()}
        self.unions = default_unions

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.seg_key not in d:
            if self.allow_missing_keys:
                return d
            raise KeyError(f"BraTSLabelToRegionsd: missing seg_key '{self.seg_key}'. Keys={list(d.keys())}")

        seg = d[self.seg_key]
        if not torch.is_tensor(seg):
            seg = torch.as_tensor(seg)

        # Ensure integer labels
        seg_int = seg.long()

        # Build base masks lazily as needed
        cache: Dict[str, torch.Tensor] = {}

        def get_base(region: str) -> torch.Tensor:
            r = region.upper()
            if r in cache:
                return cache[r]
            if r == "BACKGROUND":
                m = (seg_int == 0).to(self.dtype)
            else:
                if r not in self.label_values:
                    raise KeyError(
                        f"Unknown base region '{region}'. Known bases: {sorted(self.label_values.keys())}"
                    )
                m = (seg_int == int(self.label_values[r])).to(self.dtype)
            cache[r] = m
            return m

        def get_region(region: str) -> torch.Tensor:
            r = region.upper()
            if r in cache:
                return cache[r]
            if r in self.label_values or r == "BACKGROUND":
                return get_base(r)
            if r in self.unions:
                parts = [get_base(p) for p in self.unions[r]]
                m = torch.zeros_like(parts[0])
                for p in parts:
                    m = torch.logical_or(m > 0.5, p > 0.5)
                m = m.to(self.dtype)
                cache[r] = m
                return m
            raise KeyError(f"Unknown region '{region}'. Known: bases={sorted(self.label_values)}, unions={sorted(self.unions)}")

        for r in self.regions_out:
            d[r] = get_region(r)

        if self.drop_seg:
            d.pop(self.seg_key, None)
        return d


class RandomSelectMaskAsImaged(Transform):
    """Randomly select one of multiple region masks and copy to `out_key`.

    This is used to adapt segmentation-mask training to the existing VAEModule,
    which expects `batch['image']`.

    Args:
        source_keys: Keys of candidate region masks (e.g. ["WT","TC","ET"]).
        out_key: Destination key for the selected mask (default: "image").
        choice_key: If provided, store the selected key name in this field.
        weights: Optional sampling weights aligned to `source_keys`.
        invert_prob: Probability to invert within brain mask.
        drop_keys: Optional list of keys to remove after selection.
        allow_missing_keys: If True, skip if none of the source keys are present.
    """

    def __init__(
        self,
        *,
        source_keys: Sequence[str],
        out_key: str = "image",
        choice_key: Optional[str] = "mask_kind",
        weights: Optional[Sequence[float]] = None,
        drop_keys: Optional[Sequence[str]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        self.source_keys = [str(k) for k in source_keys]
        if not self.source_keys:
            raise ValueError("RandomSelectMaskAsImaged: source_keys must be non-empty")
        self.out_key = str(out_key)
        self.choice_key = str(choice_key) if choice_key else None
        self.drop_keys = [str(k) for k in drop_keys] if drop_keys else None
        self.allow_missing_keys = bool(allow_missing_keys)

        if weights is not None:
            if len(weights) != len(self.source_keys):
                raise ValueError("RandomSelectMaskAsImaged: weights must match source_keys length")
            w = torch.tensor([float(x) for x in weights])
            if torch.any(w < 0):
                raise ValueError("RandomSelectMaskAsImaged: weights must be non-negative")
            if float(w.sum()) == 0.0:
                raise ValueError("RandomSelectMaskAsImaged: weights sum to zero")
            self.weights = (w / w.sum()).tolist()
        else:
            self.weights = None

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        d = dict(data)

        present = [k for k in self.source_keys if k in d]
        if not present:
            if self.allow_missing_keys:
                return d
            raise KeyError(
                f"RandomSelectMaskAsImaged: none of source_keys present. Want={self.source_keys} Keys={list(d.keys())}"
            )

        # Choose among *configured* source keys, but only those present.
        # If weights are provided, renormalize over present keys.
        if self.weights is None or len(present) != len(self.source_keys):
            # uniform over present
            idx = int(torch.randint(low=0, high=len(present), size=(1,)).item())
            chosen = present[idx]
        else:
            w = torch.tensor(self.weights)
            idx = int(torch.multinomial(w, 1).item())
            chosen = self.source_keys[idx]

        mask = _as_float_tensor(d[chosen])

        d[self.out_key] = (mask > 0.5).float()
        if self.choice_key:
            d[self.choice_key] = str(chosen)

        if self.drop_keys:
            for k in self.drop_keys:
                if k != self.out_key and k != (self.choice_key or ""):
                    d.pop(k, None)

        return d


class RandForegroundBBoxCropd(Transform):
    """Crop a fixed-size 3D patch near/overlapping the foreground of a binary mask.

    Intended use: **after** selecting a single binary region mask into
    `batch[out_key]` (typically `batch["image"]`). This transform then crops a
    patch that is guaranteed to be *near* the selected region.

    Sampling rule (per spatial axis):
      - Let [imin, imax] be the min/max foreground indices along that axis.
      - Choose a crop start `s` uniformly from:
            s ∈ [max(0, imin - P + 1), min(L - P, imax)]
        which guarantees the patch [s, s+P) overlaps foreground when possible.

    If the foreground is empty, we fall back to a uniform random crop.

    Notes:
      - This keeps tensor shapes fixed (important for collation).
      - If you use cubic patch_size (e.g. 48^3 or 64^3), 90-degree rotations
        will not change the shape.

    Args:
        keys: Keys to crop (all cropped with the same ROI). Usually ["image"].
        patch_size: Spatial patch size (Px, Py, Pz).
        ref_key: Key used to compute the foreground bbox. Default: keys[0].
        allow_missing_keys: If True, skip keys that are missing.
        fg_threshold: Foreground threshold; voxels > fg_threshold are foreground.
        pad_if_needed: If True, pad tensors with zeros so each spatial dim >= patch_size.
    """

    def __init__(
        self,
        *,
        keys: Sequence[str] = ("image",),
        patch_size: Sequence[int] = (64, 64, 64),
        ref_key: Optional[str] = None,
        allow_missing_keys: bool = False,
        fg_threshold: float = 0.5,
        pad_if_needed: bool = False,
    ) -> None:
        self.keys = [str(k) for k in keys]
        if not self.keys:
            raise ValueError("RandForegroundBBoxCropd: keys must be non-empty")

        ps = [int(x) for x in patch_size]
        if len(ps) != 3 or any(p <= 0 for p in ps):
            raise ValueError(f"RandForegroundBBoxCropd: patch_size must be 3 positive ints, got {patch_size}")
        self.patch_size = tuple(ps)

        self.ref_key = str(ref_key) if ref_key is not None else self.keys[0]
        self.allow_missing_keys = bool(allow_missing_keys)
        self.fg_threshold = float(fg_threshold)
        self.pad_if_needed = bool(pad_if_needed)

    def _maybe_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-pad spatial dims to at least patch_size."""
        if not self.pad_if_needed:
            return x

        # Expect (C, X, Y, Z) or (B, C, X, Y, Z)
        if x.dim() == 4:
            spatial = x.shape[1:]
            pad = []
            for L, P in zip(reversed(spatial), reversed(self.patch_size)):
                if L >= P:
                    pad.extend([0, 0])
                else:
                    diff = P - L
                    # pad only at the end for simplicity
                    pad.extend([0, diff])
            if any(pad):
                x = F.pad(x, pad, mode="constant", value=0.0)
            return x

        if x.dim() == 5:
            spatial = x.shape[2:]
            pad = []
            for L, P in zip(reversed(spatial), reversed(self.patch_size)):
                if L >= P:
                    pad.extend([0, 0])
                else:
                    diff = P - L
                    pad.extend([0, diff])
            if any(pad):
                x = F.pad(x, pad, mode="constant", value=0.0)
            return x

        raise ValueError(f"RandForegroundBBoxCropd: expected 4D or 5D tensor, got shape {tuple(x.shape)}")

    def _compute_start(self, mask_spatial: torch.Tensor) -> Sequence[int]:
        """Compute crop start indices based on foreground bbox of a 3D spatial mask."""
        # mask_spatial: (X, Y, Z) boolean
        X, Y, Z = mask_spatial.shape
        Px, Py, Pz = self.patch_size

        if Px > X or Py > Y or Pz > Z:
            raise ValueError(
                f"RandForegroundBBoxCropd: patch_size={self.patch_size} exceeds tensor spatial={mask_spatial.shape}. "
                "Set pad_if_needed=True or reduce patch_size."
            )

        coords = torch.nonzero(mask_spatial, as_tuple=False)
        if coords.numel() == 0:
            # empty foreground: uniform random crop
            sx = int(torch.randint(0, X - Px + 1, (1,)).item())
            sy = int(torch.randint(0, Y - Py + 1, (1,)).item())
            sz = int(torch.randint(0, Z - Pz + 1, (1,)).item())
            return (sx, sy, sz)

        mins = coords.min(dim=0).values
        maxs = coords.max(dim=0).values
        (xmin, ymin, zmin) = [int(v.item()) for v in mins]
        (xmax, ymax, zmax) = [int(v.item()) for v in maxs]

        def choose(L: int, P: int, imin: int, imax: int) -> int:
            low = max(0, imin - P + 1)
            high = min(L - P, imax)
            if low > high:
                # Degenerate range: clamp near foreground
                s = min(max(0, imin), L - P)
                return int(s)
            return int(torch.randint(low, high + 1, (1,)).item())

        sx = choose(X, Px, xmin, xmax)
        sy = choose(Y, Py, ymin, ymax)
        sz = choose(Z, Pz, zmin, zmax)
        return (sx, sy, sz)

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        d = dict(data)

        if self.ref_key not in d:
            if self.allow_missing_keys:
                return d
            raise KeyError(f"RandForegroundBBoxCropd: missing ref_key '{self.ref_key}'. Keys={list(d.keys())}")

        ref = d[self.ref_key]
        if not torch.is_tensor(ref):
            ref = torch.as_tensor(ref)

        ref = _as_float_tensor(ref)
        ref = self._maybe_pad(ref)

        # Build a spatial foreground mask (union over channels / batch)
        if ref.dim() == 4:
            # (C, X, Y, Z) -> (X, Y, Z)
            fg = (ref > self.fg_threshold).any(dim=0)
        elif ref.dim() == 5:
            # (B, C, X, Y, Z) -> (X, Y, Z)
            fg = (ref > self.fg_threshold).any(dim=(0, 1))
        else:
            raise ValueError(f"RandForegroundBBoxCropd: expected 4D or 5D ref tensor, got shape {tuple(ref.shape)}")

        sx, sy, sz = self._compute_start(fg)
        Px, Py, Pz = self.patch_size
        xs = slice(sx, sx + Px)
        ys = slice(sy, sy + Py)
        zs = slice(sz, sz + Pz)

        for k in self.keys:
            if k not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"RandForegroundBBoxCropd: missing key '{k}'. Keys={list(d.keys())}")
            x = d[k]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            x = _as_float_tensor(x)
            x = self._maybe_pad(x)

            if x.dim() == 4:
                d[k] = x[:, xs, ys, zs]
            elif x.dim() == 5:
                d[k] = x[:, :, xs, ys, zs]
            else:
                raise ValueError(f"RandForegroundBBoxCropd: expected 4D/5D tensor for key '{k}', got shape {tuple(x.shape)}")

        return d
