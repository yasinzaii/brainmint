from typing import Mapping, Optional

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.diffusion.samplers.base import TimestepSamplerBase


class DiffusionUNetTimestepSampler(TimestepSamplerBase):
    """Sampler for :class:`brainmint.models.generation.diffusion_unet.DiffusionUNet`.

    Supports class conditioning and optional demographics conditioning.

    Conditioning keys (by default):
      - ``class_labels``: (B,) long
      - ``demographics_embedding``: (B, D) float (optional)
      - ``context``: (B, 1, C_ctx) float (optional; cross-attn)
    """

    def __init__(
        self,
        *,
        num_inference_steps: int = 50,
        class_labels_key: str = "class_labels",
        demographics_key: str = "demographics_embedding",
        context_key: str = "context",
    ) -> None:
        super().__init__(num_inference_steps=num_inference_steps)
        self.class_labels_key = str(class_labels_key)
        self.demographics_key = str(demographics_key)
        self.context_key = str(context_key)

    def predict_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        conditioning: Mapping[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> torch.Tensor:
        unet = ctx.get("unet", required=True)

        class_labels: Optional[torch.Tensor] = conditioning.get(self.class_labels_key)
        if class_labels is not None:
            if class_labels.ndim != 1:
                raise ValueError(f"{self.class_labels_key} must have shape (B,), got {tuple(class_labels.shape)}")
            class_labels = class_labels.to(device=ctx.device, dtype=torch.long)

        demographics: Optional[torch.Tensor] = conditioning.get(self.demographics_key)
        if demographics is not None:
            if demographics.ndim != 2:
                raise ValueError(f"{self.demographics_key} must have shape (B,D), got {tuple(demographics.shape)}")
            demographics = demographics.to(device=ctx.device, dtype=x.dtype)

        context: Optional[torch.Tensor] = conditioning.get(self.context_key)
        if context is not None:
            context = context.to(device=ctx.device, dtype=x.dtype)

        # MONAI-style UNet expects timesteps as LongTensor
        t = t.to(device=ctx.device).long()

        # Our DiffusionUNet signature is compatible with these kwargs.
        return unet(
            x=x.to(device=ctx.device),
            timesteps=t,
            context=context,
            class_labels=class_labels,
            demographics_embedding=demographics,
        )
