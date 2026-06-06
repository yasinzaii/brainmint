from __future__ import annotations

from typing import Mapping

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.diffusion.samplers.base import TimestepSamplerBase


class UkbLdmDdimSampler(TimestepSamplerBase):
    """DDIM sampler adapter for the released UKB-LDM UNet signature."""

    def __init__(self, num_inference_steps: int = 50):
        super().__init__(num_inference_steps=num_inference_steps)

    def predict_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        conditioning: Mapping[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> torch.Tensor:
        unet = ctx.get("unet", required=True)

        context = conditioning.get("conditioning")
        if context is None:
            context = conditioning.get("context")
        if context is None:
            raise KeyError("UKB-LDM sampler requires conditioning['conditioning'] or conditioning['context'] (B,1,4)")
        context = context.to(device=ctx.device, dtype=x.dtype)

        cond_concat = conditioning.get("cond_concat")
        if cond_concat is None:
            if context.ndim != 3 or context.shape[1] != 1 or context.shape[2] != 4:
                raise ValueError(f"Expected context shape (B,1,4) for UKB-LDM, got {tuple(context.shape)}")
            vec = context.squeeze(1)
            b, c = vec.shape
            z, y, xw = (int(size) for size in x.shape[-3:])
            cond_concat = vec.view(b, c, 1, 1, 1).expand(b, c, z, y, xw)
        cond_concat = cond_concat.to(device=ctx.device, dtype=x.dtype)

        if cond_concat.shape[0] != x.shape[0]:
            raise ValueError(f"cond_concat batch mismatch: {cond_concat.shape[0]} vs {x.shape[0]}")
        if cond_concat.shape[-3:] != x.shape[-3:]:
            raise ValueError(f"cond_concat spatial mismatch: {cond_concat.shape[-3:]} vs {x.shape[-3:]}")

        model_input = torch.cat((x, cond_concat), dim=1)
        return unet(x=model_input, timesteps=t.to(device=ctx.device).long(), context=context)
