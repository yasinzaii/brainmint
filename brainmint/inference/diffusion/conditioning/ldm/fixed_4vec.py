from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.diffusion.conditioning.base import ConditioningBuilderBase
from brainmint.inference.diffusion.conditioning.common.vector_ops import as_2d, expand_to_spatial, to_context


class UkbLdmFixed4VecConditioning(ConditioningBuilderBase):
    """
    UKB-LDM fixed 4-float conditioning.

    This matches MONAI UKB-LDM inference pattern where:
      - cross-attn context is a single-token tensor of shape (B, 1, 4)
      - concat conditioning channels are spatial broadcasts of those 4 floats: (B, 4, Z, Y, X)
    """

    def __init__(self, values: Sequence[float]):
        super().__init__()
        vals = [float(v) for v in list(values)]
        self.register_buffer("_values", torch.tensor(vals, dtype=torch.float32), persistent=False)

    def build(
        self, batch: Mapping[str, Any], *, latent_ref: torch.Tensor, ctx: InferenceContext
    ) -> Dict[str, torch.Tensor]:
        b = int(latent_ref.shape[0])
        spatial: Tuple[int, int, int] = tuple(int(s) for s in latent_ref.shape[-3:])

        vec = self._values.to(device=ctx.device, dtype=ctx.dtype).view(1, 4).expand(b, 4)
        vec = as_2d(vec, name="ukb_fixed_4vec")

        conditioning = to_context(vec)  # (B,1,4)
        cond_concat = expand_to_spatial(vec, spatial)  # (B,4,Z,Y,X)

        return {
            "cond": vec,
            "conditioning": conditioning,
            "context": conditioning,
            "cond_concat": cond_concat,
        }
