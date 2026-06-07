from __future__ import annotations

import inspect
from typing import Any

import torch


def set_timesteps_safe(
    scheduler: Any,
    num_steps: int,
    *,
    device: torch.device,
    input_img_size_numel: int | None = None,
) -> torch.Tensor:
    """Call ``scheduler.set_timesteps`` across MONAI / diffusers-like variants.

    Different scheduler implementations expose different signatures. We inspect parameters and
    pass what is supported.
    """
    if not hasattr(scheduler, "set_timesteps"):
        raise AttributeError("noise_scheduler missing set_timesteps()")

    sig = inspect.signature(scheduler.set_timesteps)
    kwargs = {}
    if "num_inference_steps" in sig.parameters:
        kwargs["num_inference_steps"] = int(num_steps)
    elif "num_steps" in sig.parameters:
        kwargs["num_steps"] = int(num_steps)
    else:
        # Some schedulers take a single positional arg
        kwargs = None

    if kwargs is not None:
        if "device" in sig.parameters:
            kwargs["device"] = device

        # MONAI rectified-flow schedulers require extra input size info for timestep transforms.
        if "input_img_size_numel" in sig.parameters:
            if input_img_size_numel is None:
                if bool(getattr(scheduler, "use_timestep_transform", False)):
                    raise ValueError(
                        "noise_scheduler.set_timesteps requires input_img_size_numel when use_timestep_transform=True. "
                        "Pass input_img_size_numel (e.g. x[0].numel()) or disable use_timestep_transform."
                    )
            else:
                kwargs["input_img_size_numel"] = int(input_img_size_numel)

        scheduler.set_timesteps(**kwargs)
    else:
        scheduler.set_timesteps(int(num_steps))

    timesteps = getattr(scheduler, "timesteps", None)
    if timesteps is None:
        raise AttributeError("noise_scheduler has no attribute 'timesteps' after set_timesteps()")
    return timesteps.to(device)


def scheduler_step_safe(
    scheduler: Any,
    model_output: torch.Tensor,
    t: Any,
    sample: torch.Tensor,
    *,
    next_t: Any | None = None,
) -> torch.Tensor:
    """Normalize ``scheduler.step`` output to a Tensor.

    Common patterns:
      - returns a dataclass/obj with ``prev_sample``
      - returns a tuple where the first entry is the new sample
      - returns a Tensor directly
    """
    # Some schedulers (e.g. MONAI rectified-flow / RFlowScheduler) take an extra ``next_t`` argument.
    step_sig = inspect.signature(scheduler.step)
    step_params = step_sig.parameters
    wants_next = False
    if next_t is not None:
        # MONAI RFlowScheduler uses an extra next timestep argument; names vary across versions.
        if any(k in step_params for k in ("next_timestep", "next_step")):
            wants_next = True

    if wants_next:
        out = scheduler.step(model_output, t, sample, next_t)
    else:
        out = scheduler.step(model_output, t, sample)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        if not out:
            raise TypeError("scheduler.step returned empty tuple/list")
        if not isinstance(out[0], torch.Tensor):
            raise TypeError(f"scheduler.step tuple[0] is not Tensor: {type(out[0])}")
        return out[0]
    prev = getattr(out, "prev_sample", None)
    if isinstance(prev, torch.Tensor):
        return prev
    raise TypeError(f"Unsupported scheduler.step return type: {type(out)}")
