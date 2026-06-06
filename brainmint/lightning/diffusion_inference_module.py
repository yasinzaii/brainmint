from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import pytorch_lightning as pl
import torch
from torch import nn

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import DiffusionPipeline
from brainmint.utils.state_dict_loader import StateDictLoaderMixin

log = logging.getLogger(__name__)


class DiffusionInferenceModule(StateDictLoaderMixin, pl.LightningModule):
    """Legacy inference-only Lightning shell for diffusion pipelines.

    This class is retained for older Hydra configs that still use a Lightning
    wrapper to load weights and call a diffusion pipeline. New inference code
    should use :class:`brainmint.inference.core.runner.ContextPipelineRunner`.

    It builds an :class:`~brainmint.inference.core.context.InferenceContext` and
    passes it to the pipeline, avoiding hardcoded optional pieces such as
    autoencoders, embedders, and scale factors.

    Conventional module keys inserted into context:
      - ``unet``, ``noise_scheduler``, ``autoencoder`` (if provided)
      - any extra modules you provide via ``extra_modules`` (e.g. demographics embedder)
    """


    def __init__(
        self,
        *,
        pipeline: DiffusionPipeline,
        diffusion_unet: nn.Module,
        noise_scheduler: Any,
        autoencoder: Optional[nn.Module] = None,
        extra_modules: Optional[Dict[str, Any]] = None,
        scalars: Optional[Dict[str, Any]] = None,
        hparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(dict(hparams or {}), ignore=["pipeline", "diffusion_unet", "noise_scheduler", "autoencoder"])

        self.pipeline = pipeline
        self.unet = diffusion_unet
        self.noise_scheduler = noise_scheduler
        self.autoencoder = autoencoder

        # Extra modules (e.g. demographics encoder) are part of the inference context. 
        # When an extra is an ``nn.Module``, we also register it as a proper
        # Lightning submodule so that:
        #   - ``.to(device)`` moves it with the module,
        #   - weights can be loaded via StateDictLoaderMixin (targets can reference the key).
        # Non-modules (tokenizers, callables, configs) remain in the context only.
        self._extra_modules: Dict[str, Any] = dict(extra_modules or {})
        for k, v in list(self._extra_modules.items()):
            if isinstance(v, nn.Module):
                if hasattr(self, k):
                    raise ValueError(
                        f"extra_modules contains key '{k}' which collides with an existing attribute on "
                        f"{type(self).__name__}. Choose a different key."
                    )
                setattr(self, k, v)
        self._scalars: Dict[str, Any] = dict(scalars or {})

    def setup(self, stage: Optional[str] = None) -> None:
        # StateDictLoaderMixin hooks into Lightning's setup lifecycle.
        super().setup(stage)
        self._validate_requirements()

    def _validate_requirements(self) -> None:
        """Fail fast if required context keys are missing."""
        # Build a context using current device/dtype, but avoid depending on a batch.
        ctx = self.build_context()
        required = set()
        if hasattr(self.pipeline, "get_required_modules"):
            required |= self.pipeline.get_required_modules()
        missing = [k for k in sorted(required) if ctx.modules.get(k) is None]
        if missing:
            raise KeyError(f"Missing required modules for pipeline: {missing}. Available: {sorted(ctx.modules.keys())}")

    def build_context(self, *, device: Optional[torch.device] = None) -> InferenceContext:
        dev = self.device if device is None else device
        # Prefer UNet param dtype if available; otherwise fall back to float32.
        try:
            dtype = next(self.unet.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

        modules: Dict[str, Any] = {
            "unet": self.unet,
            "noise_scheduler": self.noise_scheduler,
        }
        if self.autoencoder is not None:
            modules["autoencoder"] = self.autoencoder
        
        # Include extras exactly as provided (including any non-modules).
        modules.update(self._extra_modules)

        return InferenceContext(device=dev, dtype=dtype, modules=modules, scalars=self._scalars)

    @torch.no_grad()
    def run(self, batch: Mapping[str, Any], *, device: Optional[torch.device] = None) -> Dict[str, Any]:
        ctx = self.build_context(device=device)
        return self.pipeline.run(batch, ctx=ctx)
