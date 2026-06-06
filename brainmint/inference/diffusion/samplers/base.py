from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import DiffusionSampler, LatentInput
from brainmint.inference.core.scheduler import scheduler_step_safe, set_timesteps_safe


class TimestepSamplerBase(DiffusionSampler):
    """Generic scheduler timestep loop.

    Subclasses implement :meth:`predict_model_output` which defines how to call the UNet for the
    given model family (input construction, signature differences, extra embeddings, CFG, etc.).
    """

    required_modules = {"unet", "noise_scheduler"}

    def __init__(self, *, num_inference_steps: int = 50, init_noise_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.num_inference_steps = int(num_inference_steps)
        self.init_noise_dtype = init_noise_dtype

    @abstractmethod
    def predict_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        conditioning: Mapping[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> torch.Tensor:
        """Return model output for the current sample x and timestep t (shape like x)."""
        raise NotImplementedError

    @torch.no_grad()
    def sample_latent(
        self,
        *,
        latent: LatentInput,
        conditioning: Mapping[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> Dict[str, Any]:
        scheduler = ctx.get("noise_scheduler", required=True)

        ref = latent.ref
        init = latent.init

        if init is not None:
            x = init.to(device=ctx.device, dtype=self.init_noise_dtype)
            b = int(init.shape[0])
        elif ref is not None:
            x = torch.randn(ref.shape, device=ctx.device, dtype=self.init_noise_dtype)
            b = int(ref.shape[0])
        else:
            raise ValueError("LatentInput must provide 'init' or 'ref' for sampling.")

        # Some schedulers (e.g. MONAI rectified-flow) need extra input-size info for timestep transforms.
        # Prefer passing the true input image spatial size (not latent size) via scalars.input_img_size.
        input_img_size_numel = int(x[0].numel())
        timesteps = set_timesteps_safe(
            scheduler,
            self.num_inference_steps,
            device=ctx.device,
            input_img_size_numel=input_img_size_numel,
        )

        # Some schedulers (e.g. RFlow) require the next timestep in .step(..., next_t).
        all_t = timesteps.to(ctx.device)
        for i, t in enumerate(all_t):
            # scheduler timesteps may be scalar tensors
            t_batch = t.expand(b).to(device=ctx.device)
            model_out = self.predict_model_output(x, t_batch, conditioning=conditioning, ctx=ctx)

            t_step = int(t.item()) if torch.is_tensor(t) and t.numel() == 1 else t
            next_t = int(all_t[i + 1].item()) if (i + 1) < len(all_t) else 0
            x = scheduler_step_safe(scheduler, model_out, t_step, x, next_t=next_t)

            # latent.mask is reserved for future inpainting/editing samplers.

        return {"latent": x}
