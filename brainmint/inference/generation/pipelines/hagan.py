from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import DiffusionPipeline, Postprocessor
from brainmint.utils.batch import infer_batch_size
from brainmint.utils.spatial import center_crop_or_pad_zyx


class HaGANGenerationPipeline(DiffusionPipeline):
    """Pipeline to generate samples with HA-GAN."""

    required_modules: ClassVar[set[str]] = {"hagan"}

    def __init__(
        self,
        *,
        output_size_zyx: Sequence[int] | None = None,
        postprocess: Postprocessor | None = None,
    ) -> None:
        super().__init__()
        self.output_size_zyx = list(output_size_zyx) if output_size_zyx is not None else None
        self.postprocess = postprocess

    @torch.no_grad()
    def run(self, batch: Mapping[str, Any], *, ctx: InferenceContext) -> dict[str, Any]:
        model = ctx.get("hagan", required=True)
        if not hasattr(model, "sample"):
            raise TypeError("Context module 'hagan' must expose a .sample(batch_size=...) method.")

        sample = model.sample(batch_size=infer_batch_size(batch), device=ctx.device, dtype=ctx.dtype)

        if self.output_size_zyx is not None and self.postprocess is None:
            sample = center_crop_or_pad_zyx(sample, self.output_size_zyx)

        if self.postprocess is not None:
            metadata = {"modality": batch["modality"]} if "modality" in batch else {}
            sample = self.postprocess(sample, ctx=ctx.with_updates(metadata=metadata))

        return {"sample": sample}
