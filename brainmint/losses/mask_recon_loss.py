

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


class MaskReconLoss(nn.Module):
    """BCEWithLogits + Dice reconstruction loss for 3D single-channel binary masks.

    Args:
        bce_weight: Weight for BCE-with-logits term.
        dice_weight: Weight for Dice term.
        pos_weight: Optional positive class weighting for BCE-with-logits. Useful
            for very small foregrounds (e.g., ET). If provided, should be > 0.
            For single-channel BCE this is a scalar/broadcastable tensor.
        smooth: Dice smoothing (applied to both numerator and denominator in MONAI).
        reduction: Reduction for BCE term ("mean" | "sum" | "none"). Dice uses the
            same reduction.
        clamp_target: If True, clamp target into [0,1] (safety for any accidental
            interpolation). Normally targets should already be binary.

    Notes:
        - `logits` and `target` are expected to be shape [B, 1, X, Y, Z].
        - DiceLoss is configured with sigmoid=True, so it receives logits.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: Optional[float] = None,
        smooth: float = 1e-5,
        reduction: str = "mean",
        clamp_target: bool = False,
    ) -> None:
        super().__init__()

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.clamp_target = bool(clamp_target)

        reduction = str(reduction).lower().strip()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"MaskReconLoss: invalid reduction='{reduction}'")
        self.reduction = reduction

        pw = None
        if pos_weight is not None:
            pw_val = float(pos_weight)
            if pw_val <= 0:
                raise ValueError("MaskReconLoss: pos_weight must be > 0")
            # For single-channel BCE, a 1-element tensor is broadcastable.
            pw = torch.as_tensor([pw_val], dtype=torch.float32)
        self.register_buffer("_pos_weight", pw, persistent=False)

        sm = float(smooth)
        # For single-channel foreground masks: include_background MUST be True.
        # (include_background=False would drop the only channel.)
        self.dice = DiceLoss(
            include_background=True,
            sigmoid=True,
            smooth_nr=sm,
            smooth_dr=sm,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if logits.dim() != target.dim():
            raise ValueError(f"MaskReconLoss: logits/target dim mismatch: {logits.dim()} vs {target.dim()}")
        if logits.dim() != 5:
            raise ValueError(f"MaskReconLoss: expected 5D tensors [B,1,X,Y,Z], got {tuple(logits.shape)}")

        # Type match
        target = target.to(dtype=logits.dtype)
        if self.clamp_target:
            target = target.clamp_(0.0, 1.0)

        # BCE-with-logits (functional) so we can safely move pos_weight to device.
        pos_w = None
        if self._pos_weight is not None:
            pos_w = self._pos_weight.to(device=logits.device, dtype=logits.dtype)

        loss_bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_w,
            reduction=self.reduction,
        )

        loss_dice = self.dice(logits, target)

        return (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)


# Backwards-compatible alias (some configs or experiments may still refer to this).
MaskBCEDiceLoss = MaskReconLoss
