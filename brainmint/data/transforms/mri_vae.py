"""
MRI-only VAE transform for BrainScape.

* No resampling (already 1 mm³ voxels, same matrix size)
* Crops / pads to a fixed 3-D window, centre-aligned
* Three affine modes:
    - "original" : none
    - "rigid"    : flips + rot90 + small rotations (±0.1 rad) + optional XY-shift
    - "rand_zoom": rigid + zoom 0.8-1.2
* Intensity mapped to [0, 1] using 0-99.6 %ile.
"""

from typing import Any

import torch
from hydra.utils import instantiate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd,
    SelectItemsd,
)
from omegaconf import DictConfig, ListConfig


def _instantiate_list(xforms_like: Any | None):
    """Instantiate a list of transforms from Hydra configs or pass through callables."""
    if xforms_like is None:
        return []
    if isinstance(xforms_like, (list, tuple, ListConfig)):
        out = []
        for x in xforms_like:
            out.append(instantiate(x) if isinstance(x, DictConfig) else x)
        return out
    return [instantiate(xforms_like) if isinstance(xforms_like, DictConfig) else xforms_like]
    

def _rand_intensity_aug(image_keys: list[str]):
    """Random intensity perturbations."""
    return [
        RandBiasFieldd(keys=image_keys, prob=0.3, coeff_range=(0.0, 0.3)),
        RandGibbsNoised(keys=image_keys, prob=0.3, alpha=(0.4, 0.8)),
        RandAdjustContrastd(keys=image_keys, prob=0.3, gamma=(0.5, 2.0)),
        RandHistogramShiftd(keys=image_keys, prob=0.08, num_control_points=10),
    ]

def get_vae_transform(
    is_train: bool,
    affine_type: str = "original",  # "original" | "rigid" | "rand_zoom"
    random_aug: bool = True,
    patch_size: tuple[int, int, int] = (256, 256, 256),
    val_patch_size: tuple[int, int, int] = (256, 256, 256),
    brain_roi_size: tuple[int, int, int] = (180, 200, 155),
    output_dtype: torch.dtype = torch.float32,
    image_keys: list[str] | None = None,
    label_keys: list[str] | None = None,
    passthrough_keys: list[str] | None = None,
    meta_keys: list[str] | None = None,

    input_image_as_conditioning = False, 

    extra_xforms_start: list[Any] | None = None,
    extra_xforms_end: list[Any] | None = None,
) -> Compose:
    """Return a Compose transform matching the requested settings.

    ``image_keys`` and ``label_keys`` undergo the heavy spatial/intensity augmentations
    while ``passthrough_keys`` are simply loaded.  All keys listed
    (including ``meta_keys``) are retained by an initial ``SelectItemsd``.
    """

    image_keys = ["image"] if image_keys is None else image_keys
    label_keys = [] if label_keys is None else label_keys
    passthrough_keys = [] if passthrough_keys is None else passthrough_keys
    meta_keys = [] if meta_keys is None else meta_keys

    img_lbl_keys = list(image_keys) + list(label_keys)
    all_keys = img_lbl_keys + list(passthrough_keys) + list(meta_keys)

    load_keys = img_lbl_keys + list(passthrough_keys)

    # TODO: Correctly Manage batches with missing label_keys ?
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    xforms: list[Any] = []

    xforms += _instantiate_list(extra_xforms_start)
    xforms += [SelectItemsd(keys=all_keys, allow_missing_keys=True)]

    if load_keys:
        xforms += [
            LoadImaged(keys=load_keys, allow_missing_keys=True),
            EnsureChannelFirstd(keys=load_keys, allow_missing_keys=True),
        ]

    if img_lbl_keys:
        xforms += [

            Orientationd(keys=img_lbl_keys, axcodes="RAS", allow_missing_keys=True),
            
            # Removing around brain_roi_size of outer regions
            ResizeWithPadOrCropd(
                keys=img_lbl_keys,
                spatial_size=brain_roi_size,
                allow_missing_keys=True,
            ),
        ]

        # Initial normalisation (0–99.6 %ile → 0-1)
        if(image_keys):
            xforms += [
                ScaleIntensityRangePercentilesd(
                    keys=image_keys,
                    allow_missing_keys=True,
                    lower=0.2,
                    upper=99.6,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                )
            ]
        
        # Skipping - using ScaleIntensityRangePercentilesd
        # xforms += [ScaleIntensityd(
        #     keys=image_keys,
        #     minv = 0.0,
        #     maxv = 1.0,
        # )]

        # Stochastic intensity aug (training only)
        if is_train and random_aug:
            xforms += _rand_intensity_aug(image_keys)
            xforms += [
                RandScaleIntensityd(
                    keys=image_keys,
                    allow_missing_keys=True,
                    prob=0.3,
                    factors=(0.9, 1.1),
                ),
                RandShiftIntensityd(
                    keys=image_keys,
                    allow_missing_keys=True,
                    prob=0.3,
                    offsets=0.05,
                ),
            ]

        # Affine perturbations (optional)
        if is_train and affine_type != "original":
            # Translation
            min_r = max_r = 45
            xforms_rigid = [
                RandAffined(
                    keys=img_lbl_keys,
                    allow_missing_keys=True,
                    prob=0.5,
                    padding_mode="zeros",
                    translate_range=((-min_r, max_r), (-min_r, max_r), (-min_r, max_r)),
                    mode=interp_mode,
                )
            ]
            xforms_rigid += [
                RandRotate90d(
                    keys=img_lbl_keys,
                    allow_missing_keys=True,
                    prob=0.5,
                    spatial_axes=axes,
                )
                for axes in [(0, 1), (1, 2), (0, 2)]
            ]
            xforms_rigid += [
                RandFlipd(
                    keys=img_lbl_keys,
                    allow_missing_keys=True,
                    prob=0.5,
                    spatial_axis=axis,
                )
                for axis in range(3)
            ]
            xforms_rigid += [
                RandRotated(
                    keys=img_lbl_keys,
                    prob=0.5,
                    range_x=0.8,
                    range_y=0.8,
                    range_z=0.8,
                    keep_size=True,
                    mode=interp_mode,
                ),
            ]
            if affine_type == "rigid":
                xforms += xforms_rigid
            elif affine_type == "rand_zoom":
                xforms += xforms_rigid
                xforms += [
                    RandZoomd(
                        keys=img_lbl_keys,
                        prob=0.3,
                        min_zoom=0.8,
                        max_zoom=1.2,
                        keep_size=True,
                        mode=interp_mode,
                    )
                ]

        # Crop / pad 
        if is_train and patch_size is not None:
            xforms += [
                RandSpatialCropd(
                    keys=img_lbl_keys,
                    roi_size=patch_size,
                    allow_missing_keys=True,
                    random_size=False,
                    random_center=True,
                )
            ]

        elif val_patch_size is not None:
            xforms += [
                ResizeWithPadOrCropd(
                    keys=img_lbl_keys,
                    spatial_size=val_patch_size,
                    allow_missing_keys=True,
                )
            ]

        # MRI intensity scaling (Applying at bottom to get range back to 0-1)
        # xforms += [
        #     ScaleIntensityd(
        #         keys = image_keys,
        #         minv = 0.0,
        #         maxv = 1.0,
        #     )
        # ]

    # Disabling Clamping for passthrough_keys
    # if passthrough_keys:
    #     xforms += [
    #         ScaleIntensityd(keys=passthrough_keys, minv = 0.0, maxv = 1.0)
    #     ]

    # Final type casting for all tensor keys
    cast_keys = img_lbl_keys + list(passthrough_keys)
    if cast_keys:
        xforms += [EnsureTyped(keys=cast_keys, dtype=output_dtype, allow_missing_keys=True)]

    xforms += _instantiate_list(extra_xforms_end)

    return Compose(xforms)


class VAETransform:
    def __init__(
        self,
        is_train: bool,
        random_aug: bool,
        affine_type: str = "original",
        patch_size: tuple[int,int,int] = (256, 256, 256),
        val_patch_size: tuple[int,int,int] = None,
        brain_roi_size: tuple[int,int,int]  = None,
        output_dtype: torch.dtype = torch.float32,
        image_keys: list[str] | None = None,
        label_keys: list[str] | None = None,
        passthrough_keys: list[str] | None = None,
        meta_keys: list[str] | None = None,
        extra_xforms_start: list | None = None, 
        extra_xforms_end: list | None = None,
    ):
        
        image_keys = ["image"] if image_keys is None else image_keys
        label_keys = [] if label_keys is None else label_keys
        passthrough_keys = [] if passthrough_keys is None else passthrough_keys
        meta_keys = ["modality"] if meta_keys is None else meta_keys

        if affine_type not in {"original", "rigid", "rand_zoom"}:
            raise ValueError(
                "affine_type must be 'original', 'rigid' or 'rand_zoom'. "
                f"Got {affine_type}."
            )
        
        self.transform = get_vae_transform(
            is_train=is_train,
            random_aug=random_aug,
            affine_type=affine_type, 
            patch_size=patch_size,
            val_patch_size=val_patch_size,
            brain_roi_size=brain_roi_size,
            output_dtype=output_dtype,
            image_keys=image_keys,
            label_keys=label_keys,
            passthrough_keys=passthrough_keys,
            meta_keys=meta_keys,
            extra_xforms_start=extra_xforms_start,
            extra_xforms_end=extra_xforms_end,
        )
        self.is_train = is_train

        
    def __call__(self, img: dict) -> dict:
        return self.transform(img)
