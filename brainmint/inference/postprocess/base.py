
import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import Postprocessor


class IdentityPostprocess(Postprocessor):
    """No-op postprocessor."""

    def process(self, x: torch.Tensor, *, ctx: InferenceContext | None = None) -> torch.Tensor:
        return x
