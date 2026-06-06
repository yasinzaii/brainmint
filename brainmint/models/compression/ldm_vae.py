from __future__ import annotations

"""BrainMint-facing LDM-VAE compression wrapper."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import OmegaConf
from torch import nn

from brainmint.utils.state_dict_loader import load_module_state_dict


class LDMVAE(nn.Module):
    """MONAI LDM AutoencoderKL loaded as a BrainMint compression model."""

    def __init__(
        self,
        *,
        ckpt_path: str | Path,
        state_key: Optional[str] = "autoencoder",
        loader: Optional[str] = None,
        strict: bool | str = True,
        freeze: bool = True,
        set_eval: bool = True,
        autoencoder_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()

        from monai.networks.nets import AutoencoderKL

        if autoencoder_kwargs is None:
            model_kwargs = {}
        elif OmegaConf.is_config(autoencoder_kwargs):
            model_kwargs = OmegaConf.to_container(autoencoder_kwargs, resolve=True)
        else:
            model_kwargs = dict(autoencoder_kwargs)
        self.model = AutoencoderKL(**dict(model_kwargs))

        load_module_state_dict(
            self.model,
            path=str(ckpt_path),
            state_key=state_key,
            loader=loader,
            strict=strict,
            freeze=freeze,
            set_eval=set_eval,
            target_name="ldm_vae",
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        output = self.model(x, *args, **kwargs)
        return output[0] if isinstance(output, (tuple, list)) else output

    def reconstruct(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward(x, *args, **kwargs)


__all__ = ["LDMVAE"]
