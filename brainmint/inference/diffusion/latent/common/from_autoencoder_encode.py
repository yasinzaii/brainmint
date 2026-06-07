from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ....core.context import InferenceContext
from ....core.interfaces import LatentInput
from ..base import LatentProviderBase


class LatentFromAutoencoderEncode(LatentProviderBase):
    """Encode an input image into latent space using an autoencoder.

    This provider is useful when:
      - you want an initialization latent close to a reference image (e.g. for editing/inpainting),
      - or you explicitly want to diffuse around an encoded seed.

    Required module:
      - ``autoencoder`` with an ``encode(image)`` method returning a latent tensor.
    """

    required_modules = {"autoencoder"}

    def __init__(self, *, image_key: str = "image") -> None:
        super().__init__()
        self.image_key = str(image_key)

    @torch.no_grad()
    def get_latent(self, batch: Mapping[str, Any], *, ctx: InferenceContext) -> LatentInput:
        if self.image_key not in batch:
            raise KeyError(f"Batch missing image key '{self.image_key}'")
        x = batch[self.image_key]
        if not torch.is_tensor(x) or x.ndim != 5:
            raise ValueError(f"Batch[{self.image_key!r}] expected 5D torch.Tensor, got {type(x)} shape={getattr(x,'shape',None)}")

        autoencoder = ctx.get("autoencoder", required=True)
        z = autoencoder.encode(x.to(device=ctx.device, dtype=ctx.dtype))
        if not torch.is_tensor(z) or z.ndim != 5:
            raise ValueError(f"autoencoder.encode must return 5D latent tensor, got {type(z)} shape={getattr(z,'shape',None)}")
        z = z.to(device=ctx.device, dtype=ctx.dtype)
        ref = torch.zeros_like(z)
        return LatentInput(ref=ref, init=z)
