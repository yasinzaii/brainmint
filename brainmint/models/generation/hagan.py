from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn

from brainmint.integrations.hagan.generator import build_hagan_generator
from brainmint.utils.state_dict_loader import load_module_state_dict


@dataclass
class HaGANConfig:
    # Architecture / sampling
    resolution: int = 256  # 256 or 128 supported by upstream repo
    mode: str = "eval"
    latent_dim: int = 1024
    channel: int = 32
    num_class: int = 0

    # Generator checkpoint
    g_ckpt_path: str = ""
    state_key: Optional[str] = "model.module"
    loader: Optional[str] = None
    strict: bool | str = True
    freeze: bool = True
    set_eval: bool = True

    # Output mapping: upstream uses tanh -> [-1, 1]
    output_range: str = "zero_one"  # "zero_one" or "tanh"


class HaGANGenerator(nn.Module):
    """BrainMint-facing HA-GAN generator wrapper."""

    def __init__(self, *, cfg: Optional[HaGANConfig] = None, **kwargs: Any) -> None:
        super().__init__()

        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        if cfg is None:
            cfg = kwargs
        elif kwargs:
            raise ValueError("Pass either cfg or HA-GAN keyword options, not both.")

        if isinstance(cfg, dict):
            cfg = HaGANConfig(**cfg)
        elif not isinstance(cfg, HaGANConfig):
            raise TypeError(f"HA-GAN cfg must be a dict/DictConfig/HaGANConfig, got: {type(cfg)}")
        if not cfg.g_ckpt_path:
            raise ValueError("HA-GAN generator checkpoint path is required: g_ckpt_path")
        self.cfg = cfg

        self.model = build_hagan_generator(
            resolution=int(self.cfg.resolution),
            mode=str(self.cfg.mode),
            latent_dim=int(self.cfg.latent_dim),
            channel=int(self.cfg.channel),
            num_class=int(self.cfg.num_class),
        )
        load_module_state_dict(
            self.model,
            path=str(self.cfg.g_ckpt_path),
            state_key=self.cfg.state_key,
            loader=self.cfg.loader,
            strict=self.cfg.strict,
            freeze=self.cfg.freeze,
            set_eval=self.cfg.set_eval,
            target_name="hagan_generator",
        )

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        class_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype

        z = torch.randn(batch_size, int(self.cfg.latent_dim), device=device, dtype=dtype)

        if int(self.cfg.num_class) > 0:
            if class_label is None:
                raise ValueError("HA-GAN num_class > 0 but no class_label provided.")
            out = self.model(z, class_label=class_label)
        else:
            out = self.model(z)

        if out.dim() == 4:
            out = out.unsqueeze(1)
        if out.dim() != 5:
            raise ValueError(f"Expected HA-GAN output (B,1,Z,Y,X) or (B,Z,Y,X), got {tuple(out.shape)}")

        if str(self.cfg.output_range).lower() == "zero_one":
            out = (out + 1.0) / 2.0
        return out

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover
        return self.sample(*args, **kwargs)
