from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import LatentInput
from brainmint.inference.diffusion.samplers.base import TimestepSamplerBase
from brainmint.inference.core.scheduler import set_timesteps_safe


@dataclass(frozen=True)
class _StepSpec:
    """How to call scheduler.step(...) for this scheduler."""
    uses_next_t: bool


def _to_scalar_float(t: Any) -> float:
    """Convert a scheduler timestep element to a python float."""
    if torch.is_tensor(t):
        return float(t.item())
    return float(t)


def _make_monai_timestep_tensor(t: Any, *, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Match MONAI MAISI tutorials, generalized for batch size > 1.

    In MONAI sample.py the batch size is typically 1, so a one-element
    timestep tensor is enough. The MAISI UNet implementation expects `timesteps`
    to have shape (N,),
    where N == batch size. If N > 1 and we pass a (1,) tensor, time embeddings
    will be computed for N=1 and later concatenations with other (N, *) embeddings
    will fail.

    Therefore we create a (batch_size,) tensor filled with the scalar timestep.
    - shape: (B,)
    - dtype: float32
    """
    b = int(batch_size)
    if b <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return torch.full((b,), _to_scalar_float(t), device=device, dtype=torch.float32)


def _infer_step_spec(scheduler: Any) -> _StepSpec:
    """
    Detect whether scheduler.step expects next_t (used by rflow schedulers in MAISI).
    We avoid relying on class names and instead inspect signature, with a safe fallback.
    """
    try:
        sig = inspect.signature(scheduler.step)
        # Common MONAI ddpm: step(model_output, timestep, sample)
        # rflow variant in tutorial: step(model_output, timestep, sample, next_t)
        # Count parameters excluding self.
        params = [p for p in sig.parameters.values() if p.name != "self"]
        # Some schedulers accept **kwargs; if so, we'll try (with next_t) dynamically at runtime.
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        if has_var_kw:
            return _StepSpec(uses_next_t=True)
        # positional-or-keyword parameters count
        return _StepSpec(uses_next_t=len(params) >= 4)
    except Exception:
        # Conservative default: assume ddpm style (no next_t)
        return _StepSpec(uses_next_t=False)


class MonaiUNetTimestepSampler(TimestepSamplerBase):
    """
    MAISI/MONAI-aligned timestep sampler.

    This overrides the generic sampler loop to match MONAI MAISI tutorial logic while
    supporting batch sizes greater than one:
      - scheduler.set_timesteps(num_inference_steps=...)
      - iterate over scheduler.timesteps (and next timesteps for rflow)
      - call UNet with a timestep tensor of shape (B,)
      - call scheduler.step(model_output, t, x) OR scheduler.step(model_output, t, x, next_t)
        with t/next_t coming directly from scheduler.timesteps

    It also supports optional ControlNet residuals when present in ctx + conditioning:
      - ctx['controlnet'] and conditioning['controlnet_cond'] -> pass residuals into UNet
    """

    required_modules = {"unet", "noise_scheduler"}

    def __init__(
        self,
        *,
        num_inference_steps: int = 1000,
        init_noise_dtype: torch.dtype = torch.float32,
        # Conditioning keys (must be produced by the conditioning module, not the sampler).
        spacing_key: str = "spacing_tensor",
        top_region_index_key: str = "top_region_index_tensor",
        bottom_region_index_key: str = "bottom_region_index_tensor",
        # Optional conditioning fields
        context_key: str = "context",
        class_labels_key: str = "class_labels",
        # Optional ControlNet conditioning (matches MONAI sample.py pattern)
        controlnet_cond_key: str = "controlnet_cond",
        # If set, forces whether scheduler.step uses next_t. If None, auto-detect.
        use_next_t: Optional[bool] = None,
        # Autocast enabled (matches MAISI: torch.amp.autocast("cuda", enabled=True))
        autocast: bool = True,
    ) -> None:
        super().__init__(num_inference_steps=num_inference_steps, init_noise_dtype=init_noise_dtype)
        self.spacing_key = spacing_key
        self.top_region_index_key = top_region_index_key
        self.bottom_region_index_key = bottom_region_index_key
        self.context_key = context_key
        self.class_labels_key = class_labels_key
        self.controlnet_cond_key = controlnet_cond_key
        self._use_next_t_override = use_next_t
        self.autocast = bool(autocast)

    def _predict_model_output_monai(
        self,
        x: torch.Tensor,
        t_elem: Any,
        *,
        conditioning: Mapping[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> torch.Tensor:
        device = ctx.device
        unet = ctx.get("unet", required=True)

        # Build inputs exactly like MAISI.
        unet_inputs: dict[str, Any] = {
            "x": x,
            "timesteps": _make_monai_timestep_tensor(t_elem, batch_size=int(x.shape[0]), device=device),
        }

        # Structured conditioning
        if self.spacing_key in conditioning:
            unet_inputs["spacing_tensor"] = conditioning[self.spacing_key].to(device=device)
        if self.top_region_index_key in conditioning:
            unet_inputs["top_region_index_tensor"] = conditioning[self.top_region_index_key].to(device=device)
        if self.bottom_region_index_key in conditioning:
            unet_inputs["bottom_region_index_tensor"] = conditioning[self.bottom_region_index_key].to(device=device)

        # Optional conditioning fields
        if self.context_key in conditioning:
            unet_inputs["context"] = conditioning[self.context_key].to(device=device)
        if self.class_labels_key in conditioning:
            unet_inputs["class_labels"] = conditioning[self.class_labels_key].to(device=device)

        # Optional ControlNet residuals (MONAI sample.py)
        if (ctx.get("controlnet") is not None) and (self.controlnet_cond_key in conditioning):
            controlnet = ctx.get("controlnet", required=True)
            controlnet_cond = conditioning[self.controlnet_cond_key].to(device=device)
            # ControlNet uses same timestep tensor as UNet in sample.py.
            down_res, mid_res = controlnet(
                x=x,
                timesteps=unet_inputs["timesteps"],
                controlnet_cond=controlnet_cond,
            )
            unet_inputs["down_block_additional_residuals"] = down_res
            unet_inputs["mid_block_additional_residual"] = mid_res

        # Filter kwargs if UNet doesn't accept **kwargs.
        try:
            sig = inspect.signature(unet.forward)
            params = sig.parameters
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not accepts_kwargs:
                unet_inputs = {k: v for k, v in unet_inputs.items() if k in params}
        except (TypeError, ValueError):
            pass

        return unet(**unet_inputs)

    @torch.no_grad()
    def sample_latent(
        self,
        *,
        latent: LatentInput,
        conditioning: Mapping[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> Mapping[str, Any]:
        scheduler = ctx.get("noise_scheduler", required=True)

        # Initialize latent x
        ref = latent.ref
        init = latent.init
        if init is not None:
            x = init.to(device=ctx.device, dtype=self.init_noise_dtype)
            b = int(init.shape[0])
        elif ref is not None:
            b = int(ref.shape[0])
            x = torch.randn_like(ref, device=ctx.device, dtype=self.init_noise_dtype)
        else:
            raise ValueError("LatentInput must provide either init or ref.")

        # Set timesteps exactly like MAISI
        set_timesteps_safe(scheduler, num_steps=int(self.num_inference_steps), device=ctx.device)
        timesteps = scheduler.timesteps

        # Determine whether scheduler.step needs next_t (rflow)
        step_spec = _infer_step_spec(scheduler)
        uses_next_t = self._use_next_t_override if self._use_next_t_override is not None else step_spec.uses_next_t

        if uses_next_t:
            # Match tutorial: next timesteps = timesteps[1:] + [0]
            all_t = timesteps.to(ctx.device)
            next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype, device=all_t.device)))
            iterator = zip(all_t, next_t)
        else:
            iterator = [(t, None) for t in timesteps.to(ctx.device)]

        autocast_device = "cuda" if ctx.device.type == "cuda" else "cpu"
        with torch.amp.autocast(autocast_device, enabled=self.autocast and ctx.device.type == "cuda"):
            for t, nt in iterator:
                model_out = self._predict_model_output_monai(x, t, conditioning=conditioning, ctx=ctx)

                # Call scheduler.step exactly like tutorials.
                try:
                    if nt is None:
                        x, _ = scheduler.step(model_out, t, x)  # type: ignore[misc]
                    else:
                        x, _ = scheduler.step(model_out, t, x, nt)  # type: ignore[misc]
                except TypeError:
                    # Fallback: try without next_t if scheduler doesn't accept it.
                    x, _ = scheduler.step(model_out, t, x)  # type: ignore[misc]

        return {"latent": x}

    # Required by base class, but unused because we override sample_latent()
    def predict_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        conditioning: Mapping[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> torch.Tensor:
        # Delegate to the MONAI-style path for a representative timestep (first element).
        t_elem = t[0] if t.numel() > 0 else t
        return self._predict_model_output_monai(x, t_elem, conditioning=conditioning, ctx=ctx)
