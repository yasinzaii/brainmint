from typing import Any, Dict, Mapping

import torch
from torch import nn

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.diffusion.conditioning.base import ConditioningBuilderBase


class DemographicsEmbeddingConditioning(ConditioningBuilderBase):
    """Embed demographics for diffusion conditioning.

    Assumptions (intentional):
      - ``batch[values_key]`` is a tensor of shape ``(B, F)``.
      - ``batch[missing_key]`` is a tensor of shape ``(B, F)`` (bool).
      - ``B`` matches the current latent batch size.
      - ``ctx.modules[encoder_module_key]`` is a ``nn.Module`` with signature:
          ``encoder(demo_values=..., demo_missing=...)``.

    Output:
      - ``out_embed_key``: (B, E)
    """

    def __init__(
        self,
        *,
        values_key: str = "demo_values",
        missing_key: str = "demo_missing",
        encoder_module_key: str = "demographics_encoder",
        out_embed_key: str = "demographics_embedding",
    ) -> None:
        super().__init__()
        self.values_key = str(values_key)
        self.missing_key = str(missing_key)
        self.encoder_module_key = str(encoder_module_key)
        self.out_embed_key = str(out_embed_key)
        self.required_modules = {self.encoder_module_key}

    def build(
        self,
        batch: Mapping[str, Any],
        *,
        latent_ref: torch.Tensor,
        ctx: InferenceContext,
    ) -> Dict[str, torch.Tensor]:
        if self.values_key not in batch:
            raise KeyError(f"Batch missing key '{self.values_key}'")
        if self.missing_key not in batch:
            raise KeyError(f"Batch missing key '{self.missing_key}'")

        values = batch[self.values_key]
        missing = batch[self.missing_key]
        if not torch.is_tensor(values) or values.ndim != 2:
            raise ValueError(
                f"{self.values_key} must be a (B,F) tensor, got {type(values)} shape={getattr(values,'shape',None)}"
            )
        if not torch.is_tensor(missing) or missing.ndim != 2:
            raise ValueError(
                f"{self.missing_key} must be a (B,F) tensor, got {type(missing)} shape={getattr(missing,'shape',None)}"
            )
        if values.shape != missing.shape:
            raise ValueError(f"{self.values_key} shape {tuple(values.shape)} != {self.missing_key} shape {tuple(missing.shape)}")

        b = int(latent_ref.shape[0])
        if int(values.shape[0]) != b:
            raise ValueError(f"{self.values_key} has B={int(values.shape[0])} but latent_ref has B={b}")

        values = values.to(device=ctx.device, dtype=ctx.dtype)
        missing = missing.to(device=ctx.device)
        if missing.dtype != torch.bool:
            missing = missing.to(dtype=torch.bool)

        encoder = ctx.get(self.encoder_module_key, required=True)
        if not isinstance(encoder, nn.Module):
            raise TypeError(f"ctx.modules['{self.encoder_module_key}'] must be nn.Module, got {type(encoder)}")

        emb = encoder(demo_values=values, demo_missing=missing)
        if not torch.is_tensor(emb) or emb.ndim != 2:
            raise ValueError(
                f"demographics encoder must return 2D tensor (B,E), got {type(emb)} shape={getattr(emb,'shape',None)}"
            )
        if int(emb.shape[0]) != b:
            raise ValueError(f"demographics embedding has B={int(emb.shape[0])} but latent_ref has B={b}")

        return {self.out_embed_key: emb}
