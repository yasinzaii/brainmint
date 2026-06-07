from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from brainmint.inference.core.context import InferenceContext
from brainmint.integrations.med_ddpm.generator import build_med_ddpm_diffusion


@dataclass
class MedDDPMConfig:
    """Configuration for the Med-DDPM BraTS generator."""

    ckpt_path: str = ""

    # Native BraTS checkpoint shape: tensor order is (D,W,H), with 192 in-plane and 144 depth.
    image_size: int = 192
    depth_size: int = 144

    # Upstream UNet / diffusion settings. These must match the checkpoint.
    num_channels: int = 64
    num_res_blocks: int = 2
    timesteps: int = 250
    with_condition: bool = True
    out_channels: int = 4

    # Checkpoint loading. The public BraTS checkpoint stores EMA weights under ``ema``.
    state_key: str | None = "ema"
    loader: str | None = None
    strict: bool | str = True
    freeze: bool = True
    set_eval: bool = True


class MedDDPMGenerator(nn.Module):
    """BrainMint-facing wrapper around the official Med-DDPM BraTS sampler."""

    def __init__(self, *, cfg: MedDDPMConfig | None = None, **kwargs: Any) -> None:
        super().__init__()

        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        if cfg is None:
            cfg = kwargs
        elif kwargs:
            raise ValueError("Pass either cfg or Med-DDPM keyword options, not both.")

        if isinstance(cfg, dict):
            cfg = MedDDPMConfig(**cfg)
        elif not isinstance(cfg, MedDDPMConfig):
            raise TypeError(f"Med-DDPM cfg must be a dict/DictConfig/MedDDPMConfig, got: {type(cfg)}")
        if not cfg.ckpt_path:
            raise ValueError("Med-DDPM checkpoint path is required: ckpt_path")
        self.cfg = cfg

        self.diffusion: nn.Module | None = None
        self._weights_loaded = False

    def load_weights(self) -> None:
        """Instantiate upstream Med-DDPM components and load checkpoint weights."""

        if self._weights_loaded:
            return

        self.diffusion = build_med_ddpm_diffusion(
            ckpt_path=self.cfg.ckpt_path,
            image_size=self.cfg.image_size,
            depth_size=self.cfg.depth_size,
            num_channels=self.cfg.num_channels,
            num_res_blocks=self.cfg.num_res_blocks,
            timesteps=self.cfg.timesteps,
            with_condition=self.cfg.with_condition,
            out_channels=self.cfg.out_channels,
            state_key=self.cfg.state_key,
            loader=self.cfg.loader,
            strict=self.cfg.strict,
            freeze=self.cfg.freeze,
            set_eval=self.cfg.set_eval,
        )
        self._weights_loaded = True

    @torch.no_grad()
    def sample_all(
        self,
        *,
        condition_tensors: torch.Tensor,
        batch_size: int | None = None,
        ctx: InferenceContext | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Generate all four BraTS modalities conditioned on one-hot masks.

        ``condition_tensors`` must be shaped ``(B,4,D,W,H)`` with channels
        ``[TUMORAREA1, TUMORAREA2, TUMORAREA3, BRAINAREA]``. The upstream
        sampler returns ``(B,4,D,W,H)`` in roughly ``[-1, 1]``.
        """

        if not self._weights_loaded:
            self.load_weights()
        if self.diffusion is None:
            raise RuntimeError("Med-DDPM weights were not loaded.")

        dev = device or (ctx.device if ctx is not None else next(self.diffusion.parameters()).device)
        if next(self.diffusion.parameters()).device != dev:
            self.diffusion.to(dev)

        cond = condition_tensors.to(device=dev, dtype=torch.float32)
        if cond.dim() != 5 or cond.shape[1] != int(self.cfg.out_channels):
            raise ValueError(
                "Med-DDPM expects condition_tensors with shape (B,4,D,W,H); "
                f"got {tuple(cond.shape)}"
            )

        cond_dwh = cond.permute(0, 1, 4, 3, 2).contiguous()
        b = int(cond_dwh.shape[0])
        if batch_size is not None and b != int(batch_size):
            raise ValueError(
                f"Target batch size {batch_size} does not match condition tensors {tuple(cond_dwh.shape)}"
            )
        return self.diffusion.sample(batch_size=b, condition_tensors=cond_dwh)
