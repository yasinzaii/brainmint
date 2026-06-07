from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import DiffusionPipeline
from brainmint.utils.state_dict_loader import load_weight_specs

log = logging.getLogger(__name__)

def _first_module_parameter(modules: Mapping[str, Any]) -> nn.Parameter | None:
    for module in modules.values():
        if isinstance(module, nn.Module):
            parameter = next(module.parameters(), None)
            if parameter is not None:
                return parameter
    return None


class ContextPipelineRunner(nn.Module):
    """Non-Lightning runner for Hydra-composed inference pipelines."""

    def __init__(
        self,
        *,
        pipeline: DiffusionPipeline,
        auto_load_wrapper_weights: bool = False,
        modules: dict[str, Any] | None = None,
        scalars: dict[str, Any] | None = None,
        weight_loads: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.auto_load_wrapper_weights = bool(auto_load_wrapper_weights)
        self.weight_loads = [dict(spec) for spec in (weight_loads or [])]
        self._weights_loaded = False
        self._loaded_targets: set[str] = set()

        self._modules_dict: dict[str, Any] = dict(modules or {})
        for key, value in list(self._modules_dict.items()):
            if isinstance(value, nn.Module):
                if hasattr(self, key):
                    raise ValueError(
                        f"modules contains key '{key}' which collides with an existing attribute on "
                        f"{type(self).__name__}."
                    )
                setattr(self, key, value)

        self._scalars: dict[str, Any] = dict(scalars or {})

    def load_weights(self) -> None:
        if self._weights_loaded:
            return

        self._loaded_targets.update(load_weight_specs(self, self.weight_loads))

        if self.auto_load_wrapper_weights:
            for name, module in self._modules_dict.items():
                load_weights = getattr(module, "load_weights", None)
                if callable(load_weights):
                    log.info("Auto-loading wrapper weights for module '%s'", name)
                    load_weights()

        self._weights_loaded = True
        self._validate_requirements()

    def build_context(self, *, device: torch.device | None = None) -> InferenceContext:
        parameter = _first_module_parameter(self._modules_dict)
        dev = device or (parameter.device if parameter is not None else torch.device("cpu"))
        dtype = parameter.dtype if parameter is not None else torch.float32
        return InferenceContext(device=dev, dtype=dtype, modules=self._modules_dict, scalars=self._scalars)

    def _validate_requirements(self) -> None:
        ctx = self.build_context()
        required = set()
        if hasattr(self.pipeline, "get_required_modules"):
            required |= self.pipeline.get_required_modules()
        missing = sorted([key for key in required if ctx.get(key) is None])
        if missing:
            raise KeyError(
                f"ContextPipelineRunner missing required context keys: {missing}. "
                f"Available: {sorted(ctx.modules.keys())}"
            )

    @torch.no_grad()
    def run(self, batch: Mapping[str, Any], *, device: torch.device | None = None) -> dict[str, Any]:
        self.load_weights()
        ctx = self.build_context(device=device)
        return self.pipeline.run(batch, ctx=ctx)
