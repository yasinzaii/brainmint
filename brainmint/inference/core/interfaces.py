from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn

from brainmint.inference.core.context import InferenceContext


class InferenceComponent(nn.Module, ABC):
    """Base class for all inference components."""

    required_modules: ClassVar[set[str]] = set()

    def get_required_modules(self) -> set[str]:
        return set(self.required_modules)


@dataclass(frozen=True)
class LatentInput:
    """Container returned by :class:`LatentProvider`.

    ``ref`` defines shape/device/dtype. ``init``/``mask`` are optional for future tasks (e.g. inpainting).
    """

    ref: torch.Tensor | None = None
    init: torch.Tensor | None = None
    mask: torch.Tensor | None = None


class LatentProvider(InferenceComponent, ABC):
    @abstractmethod
    def get_latent(self, batch: Mapping[str, Any], *, ctx: InferenceContext) -> LatentInput:
        raise NotImplementedError


class ConditioningBuilder(InferenceComponent, ABC):
    @abstractmethod
    def build(self, batch: Mapping[str, Any], *, latent_ref: torch.Tensor, ctx: InferenceContext) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class DiffusionSampler(InferenceComponent, ABC):
    @abstractmethod
    def sample_latent(
        self,
        *,
        latent: LatentInput,
        conditioning: Mapping[str, torch.Tensor],
        ctx: InferenceContext,
    ) -> dict[str, Any]:
        raise NotImplementedError


class Postprocessor(InferenceComponent, ABC):
    def forward(self, x: torch.Tensor, *, ctx: InferenceContext | None = None) -> torch.Tensor:
        return self.process(x, ctx=ctx)

    @abstractmethod
    def process(self, x: torch.Tensor, *, ctx: InferenceContext | None = None) -> torch.Tensor:
        raise NotImplementedError


class DiffusionPipeline(InferenceComponent, ABC):
    @abstractmethod
    def run(self, batch: Mapping[str, Any], *, ctx: InferenceContext) -> dict[str, Any]:
        raise NotImplementedError
