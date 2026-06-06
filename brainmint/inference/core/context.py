from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch


@dataclass(frozen=True)
class InferenceContext:
    """Carries runtime info and component handles through the inference graph.

    **modules**
        A mapping from string keys to *objects* used during inference. Most entries are ``nn.Module``
        instances (UNet, autoencoder, embedders), but this intentionally also supports non-modules
        such as MONAI schedulers, tokenizers, or simple callables.

        Conventional diffusion keys (not enforced by core, but widely used):
          - ``"unet"``: diffusion model
          - ``"noise_scheduler"``: scheduler instance
          - ``"autoencoder"``: latent decoder/encoder (for latent diffusion)
          - optional extras: ``"demographics_embedder"``, ``"text_encoder"``, ``"controlnet"``, etc.

    **scalars**
        Non-module configuration values (floats/ints/strings/dicts). Typical examples:
          - ``scale_factor`` (latent scaling)
          - ``guidance_scale``
          - ``num_inference_steps`` (if you choose to drive steps via context)

    **device / dtype**
        Default device + dtype for tensors created inside components.
    """

    device: torch.device
    dtype: torch.dtype
    modules: Mapping[str, Any] = field(default_factory=dict)
    scalars: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def get(self, key: str, *, required: bool = False) -> Optional[Any]:
        v = self.modules.get(key)
        if required and v is None:
            raise KeyError(f"InferenceContext missing required key '{key}'. Available: {sorted(self.modules.keys())}")
        return v

    def scalar(self, key: str, default: Any = None) -> Any:
        return self.scalars.get(key, default)

    def with_updates(
        self,
        *,
        modules: Optional[Mapping[str, Any]] = None,
        scalars: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "InferenceContext":
        new_modules = dict(self.modules)
        if modules:
            new_modules.update(dict(modules))

        new_scalars = dict(self.scalars)
        if scalars:
            new_scalars.update(dict(scalars))

        new_meta = dict(self.metadata)
        if metadata:
            new_meta.update(dict(metadata))

        return InferenceContext(
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
            modules=new_modules,
            scalars=new_scalars,
            metadata=new_meta,
        )
