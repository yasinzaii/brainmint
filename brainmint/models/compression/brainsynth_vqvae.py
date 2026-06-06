from __future__ import annotations

"""BrainMint-facing BrainSynth VQ-VAE compression wrapper."""

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from omegaconf import OmegaConf
from torch import nn

from brainmint.integrations.brainsynth.vendor_vqvae import BaselineVQVAE
from brainmint.utils.state_dict_loader import load_module_state_dict


class BrainSynthVQVAE(nn.Module):
    """Vendored BrainSynth VQ-VAE loaded as a BrainMint compression model."""

    def __init__(
        self,
        *,
        ckpt_path: str | Path,
        state_key: Optional[str] = "network",
        strict: bool | str = True,
        freeze: bool = True,
        set_eval: bool = True,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()

        if model_kwargs is None:
            kwargs = {}
        elif OmegaConf.is_config(model_kwargs):
            kwargs = OmegaConf.to_container(model_kwargs, resolve=True)
        else:
            kwargs = dict(model_kwargs)

        self.model = BaselineVQVAE(**dict(kwargs))
        load_module_state_dict(
            self.model,
            path=str(ckpt_path),
            state_key=state_key,
            strict=strict,
            freeze=freeze,
            set_eval=set_eval,
            target_name="brainsynth_vqvae",
        )
        if set_eval:
            self.eval()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        output = self.model(x, *args, **kwargs)
        return output["reconstruction"][0]

    def reconstruct(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward(x, *args, **kwargs)


__all__ = ["BrainSynthVQVAE"]
