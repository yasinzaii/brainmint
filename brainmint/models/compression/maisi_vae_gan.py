from __future__ import annotations

"""BrainMint-facing MAISI VAE-GAN compression wrapper."""

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from brainmint.utils.state_dict_loader import load_module_state_dict


class MAISIVAEGAN(nn.Module):
    """Checkpointed MAISI autoencoder exposed as a compression model."""

    def __init__(
        self,
        *,
        autoencoder: nn.Module,
        ckpt_path: str | Path,
        state_key: Optional[str] = "autoencoder",
        loader: Optional[str] = None,
        strict: bool | str = True,
        freeze: bool = True,
        set_eval: bool = True,
    ) -> None:
        super().__init__()
        self.model = autoencoder
        load_module_state_dict(
            self.model,
            path=str(ckpt_path),
            state_key=state_key,
            loader=loader,
            strict=strict,
            freeze=freeze,
            set_eval=set_eval,
            target_name="maisi_vae_gan",
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        reconstruction, _, _ = self.model(x, *args, **kwargs)
        return reconstruction

    def reconstruct(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward(x, *args, **kwargs)

    def run_inference(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, Any, Any]:
        reconstruction, z_mu, z_sigma = self.model(batch["image"])
        return reconstruction, z_mu, z_sigma


__all__ = ["MAISIVAEGAN"]
