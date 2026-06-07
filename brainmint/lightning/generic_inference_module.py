from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import DiffusionPipeline
from brainmint.utils.state_dict_loader import StateDictLoaderMixin

log = logging.getLogger(__name__)


class GenericInferenceModule(StateDictLoaderMixin, pl.LightningModule):
    """Legacy inference-only Lightning shell for arbitrary pipelines.

    This class is kept for older Hydra configs that still instantiate a
    LightningModule for inference. New inference code should use
    :class:`brainmint.inference.core.runner.ContextPipelineRunner`, which provides
    the same context/pipeline pattern without depending on Lightning.
    """

    def __init__(
        self,
        *,
        pipeline: DiffusionPipeline,
        auto_load_wrapper_weights: bool = False,
        modules: dict[str, Any] | None = None,
        scalars: dict[str, Any] | None = None,
        hparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(dict(hparams or {}), ignore=["pipeline", "modules"])
        self.pipeline = pipeline
        self.auto_load_wrapper_weights = bool(auto_load_wrapper_weights)

        self._modules_dict: dict[str, Any] = dict(modules or {})
        # Register nn.Modules as proper submodules to participate in .to(device) and weight loading.
        for k, v in list(self._modules_dict.items()):
            if isinstance(v, nn.Module):
                if hasattr(self, k):
                    raise ValueError(
                        f"modules contains key '{k}' which collides with an existing attribute on {type(self).__name__}."
                    )
                setattr(self, k, v)

        self._scalars: dict[str, Any] = dict(scalars or {})

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        if self.auto_load_wrapper_weights:
            for _name, _mod in self._modules_dict.items():
                if hasattr(_mod, "load_weights") and callable(_mod.load_weights):
                    log.info("Auto-loading wrapper weights for module '%s'", _name)
                    _mod.load_weights()
        self._validate_requirements()

    def build_context(self, *, device: torch.device | None = None) -> InferenceContext:
        dev = device or next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return InferenceContext(device=dev, dtype=dtype, modules=self._modules_dict, scalars=self._scalars)

    def _validate_requirements(self) -> None:
        ctx = self.build_context()
        required = set()
        if hasattr(self.pipeline, "get_required_modules"):
            required |= self.pipeline.get_required_modules()
        missing = sorted([k for k in required if ctx.get(k) is None])
        if missing:
            raise KeyError(f"GenericInferenceModule missing required context keys: {missing}. Available: {sorted(ctx.modules.keys())}")

    @torch.no_grad()
    def run(self, batch: Mapping[str, Any], *, device: torch.device | None = None) -> dict[str, Any]:
        ctx = self.build_context(device=device)
        return self.pipeline.run(batch, ctx=ctx)
