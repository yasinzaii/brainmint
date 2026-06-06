from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Set, ClassVar

import torch

from brainmint.utils.batch import infer_batch_size
from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import DiffusionPipeline, Postprocessor
from brainmint.utils.spatial import center_crop_or_pad_zyx


class MAISI3DGenerationPipeline(DiffusionPipeline):
    """Generation pipeline for MAISI 3D diffusion."""

    required_modules: ClassVar[Set[str]] = {"maisi"}

    def __init__(
        self,
        *,
        output_size_zyx: Optional[Sequence[int]] = None,
        normalize_to_01: bool = True,
        postprocess: Optional[Postprocessor] = None,
    ) -> None:
        super().__init__()
        self.output_size_zyx = list(output_size_zyx) if output_size_zyx is not None else None
        self.normalize_to_01 = bool(normalize_to_01)
        self.postprocess = postprocess

    def run(self, batch: Mapping[str, Any], *, ctx: InferenceContext) -> Dict[str, Any]:
        model = ctx.get("maisi", required=True)
        if not hasattr(model, "sample"):
            raise TypeError("Context module 'maisi' must expose a .sample(...) method.")

        sample = model.sample(batch, batch_size=infer_batch_size(batch), ctx=ctx, device=ctx.device, dtype=ctx.dtype)

        if not isinstance(sample, torch.Tensor) or sample.dim() != 5:
            raise ValueError(
                "MAISI pipeline expected 5D tensor (B,1,Z,Y,X), "
                f"got {type(sample)} with shape {getattr(sample, 'shape', None)}"
            )

        if self.output_size_zyx is not None:
            sample = center_crop_or_pad_zyx(sample, self.output_size_zyx)

        if self.normalize_to_01:
            sample = sample.clamp(0.0, 1.0)

        if self.postprocess is not None:
            sample = self.postprocess(sample, ctx=ctx)

        return {"sample": sample}
