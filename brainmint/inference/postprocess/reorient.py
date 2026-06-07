from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import Postprocessor
from brainmint.utils.spatial import center_crop_or_pad_zyx


class ReorientCropPostprocess(Postprocessor):
    """Deterministic reorientation (permute/rot90/flip) + optional center crop/pad.

    This is meant for models that emit volumes in a different axis order or in-plane orientation.

    The input and output tensors are always treated as (B, C, Z, Y, X). The operations below
    re-map the *spatial* axes (Z,Y,X) only.

    Typical fix for "axial shows sagittal" symptoms: swap Z and X:
        spatial_permute: [2, 1, 0]

    If you still see a 90-degree in-plane rotation, set rot90_k to 1 or -1 with rot90_plane: "yx".
    """

    def __init__(
        self,
        *,
        roi_size_zyx: Sequence[int] | None = None,
        spatial_permute: Sequence[int] | None = None,
        rot90_k: int = 0,
        rot90_plane: str = "yx",
        flips: Mapping[str, bool] | None = None,
    ) -> None:
        super().__init__()

        self.roi_size_zyx = list(roi_size_zyx) if roi_size_zyx is not None else None

        if spatial_permute is not None:
            if len(spatial_permute) != 3:
                raise ValueError("spatial_permute must have length 3 (permute Z,Y,X)")
            perm = [int(i) for i in spatial_permute]
            if sorted(perm) != [0, 1, 2]:
                raise ValueError("spatial_permute must be a permutation of [0,1,2]")
            self.spatial_permute = perm
        else:
            self.spatial_permute = None

        self.rot90_k = int(rot90_k)
        self.rot90_plane = str(rot90_plane).lower()

        self.flips: dict[str, bool] = {"z": False, "y": False, "x": False}
        if flips:
            for k, v in flips.items():
                kk = str(k).lower()
                if kk not in self.flips:
                    raise ValueError(f"Unknown flip axis '{k}'. Allowed: z,y,x")
                self.flips[kk] = bool(v)

    def process(self, x: torch.Tensor, *, ctx: InferenceContext | None = None) -> torch.Tensor:
        # Accept (C,Z,Y,X) or (B,C,Z,Y,X)
        squeeze_b = False
        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze_b = True
        if x.dim() != 5:
            raise ValueError(f"Expected (B,C,Z,Y,X) or (C,Z,Y,X); got {tuple(x.shape)}")

        # 1) Spatial permute (Z,Y,X)
        if self.spatial_permute is not None:
            pz, py, px = self.spatial_permute
            x = x.permute(0, 1, 2 + pz, 2 + py, 2 + px).contiguous()

        # 2) Rot90 in selected plane
        k = self.rot90_k % 4
        if k:
            if self.rot90_plane == "yx":
                dims = (-2, -1)
            elif self.rot90_plane == "zx":
                dims = (-3, -1)
            elif self.rot90_plane == "zy":
                dims = (-3, -2)
            else:
                raise ValueError("rot90_plane must be one of: 'yx', 'zx', 'zy'")
            x = torch.rot90(x, k=k, dims=dims)

        # 3) Optional flips
        if self.flips.get("z", False):
            x = torch.flip(x, dims=[-3])
        if self.flips.get("y", False):
            x = torch.flip(x, dims=[-2])
        if self.flips.get("x", False):
            x = torch.flip(x, dims=[-1])

        # 4) Optional crop/pad
        if self.roi_size_zyx is not None:
            x = center_crop_or_pad_zyx(x, self.roi_size_zyx)

        if squeeze_b:
            x = x.squeeze(0)
        return x
