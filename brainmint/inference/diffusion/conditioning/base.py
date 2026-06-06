from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Set

import torch
from torch import nn

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import ConditioningBuilder


class ConditioningBuilderBase(ConditioningBuilder):
    """Convenience base class for diffusion conditioning builders."""


class ComposeConditioning(ConditioningBuilderBase):
    """Compose multiple conditioning builders into one conditioning dictionary.

    Each child builder returns ``dict[str, Tensor]``. Outputs are merged in order.
    Duplicate keys raise by default so config mistakes are visible early.
    """

    def __init__(
        self,
        builders: Sequence[ConditioningBuilder],
        *,
        allow_key_overwrite: bool = False,
    ) -> None:
        super().__init__()
        if not builders:
            raise ValueError("ComposeConditioning requires at least 1 builder")
        self.builders = nn.ModuleList(list(builders))
        self.allow_key_overwrite = bool(allow_key_overwrite)

    def get_required_modules(self) -> Set[str]:
        required: Set[str] = set()
        for builder in self.builders:
            required |= builder.get_required_modules()
        return required

    def build(
        self,
        batch: Mapping[str, Any],
        *,
        latent_ref: torch.Tensor,
        ctx: InferenceContext,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for builder in self.builders:
            current = builder.build(batch, latent_ref=latent_ref, ctx=ctx)
            if not isinstance(current, dict):
                raise TypeError(
                    f"ConditioningBuilder {type(builder).__name__} returned {type(current)}; expected dict"
                )
            for key, value in current.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"ConditioningBuilder {type(builder).__name__} produced key '{key}' "
                        f"with type {type(value)}; expected Tensor"
                    )
                if (not self.allow_key_overwrite) and (key in out):
                    raise KeyError(
                        f"ComposeConditioning key collision on '{key}'. "
                        "Set allow_key_overwrite=True if you really want last-writer-wins."
                    )
                out[key] = value
        return out
