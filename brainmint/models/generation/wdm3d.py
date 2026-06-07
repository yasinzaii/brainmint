from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from brainmint.inference.core.context import InferenceContext
from brainmint.integrations.wdm3d.generator import build_wdm3d_components


@dataclass
class WDM3DConfig:
    # checkpoint
    ckpt_path: str = ""

    # Which set of args to use. Matches wdm-3d run.sh blocks.
    # For pfriedri/wdm-3d BraTS checkpoint: use "brats_unet_128".
    preset: str = "brats_unet_128"

    # Sampling
    clip_denoised: bool = True
    seed: int | None = None

    # Shape & wavelet parameters
    image_size: int = 128          # output size after IDWT (e.g., 128)
    wavelet_downsample: int = 2    # wavelet domain uses image_size//2
    wavelet_components: int = 8    # 3D Haar has 8 sub-bands
    wavelet: str = "haar"
    lll_scale: float = 3.0         # as in their scripts/generation_sample.py

    # Checkpoint loading
    state_key: str | None = "<root>"
    loader: str | None = None
    strict: bool | str = True
    freeze: bool = True
    set_eval: bool = True

    # Optional overrides for create_model_and_diffusion kwargs
    # (use with care; must match checkpoint architecture)
    model_kwargs: dict[str, Any] | None = None


class WDM3DWrapper(nn.Module):
    """Wavelet Diffusion Model (WDM-3D) wrapper.

    This wrapper:
      1) samples in wavelet space (8 channels, size=image_size//2),
      2) reconstructs the voxel volume via IDWT_3D,
      3) returns (B,1,Z,Y,X) in [-1, 1] (pipeline can normalize to [0,1]).
    """
    def __init__(self, cfg: Any) -> None:
        super().__init__()

        # Hydra passes DictConfig; normalize to a plain dict then dataclass.
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        if isinstance(cfg, dict):
            cfg = WDM3DConfig(**cfg)
        elif not isinstance(cfg, WDM3DConfig):
            raise TypeError(
                f"WDM3DWrapper cfg must be a dict/DictConfig/WDM3DConfig, got: {type(cfg)}"
            )

        self.cfg: WDM3DConfig = cfg

        # Will be populated in load_weights()
        self.model: nn.Module | None = None
        self.diffusion: Any = None
        self.idwt: nn.Module | None = None

        self._weights_loaded: bool = False

    def load_weights(self) -> None:
        """Instantiate upstream WDM-3D components and load checkpoint weights."""
        if self._weights_loaded:
            return

        components = build_wdm3d_components(
            ckpt_path=self.cfg.ckpt_path,
            preset=self.cfg.preset,
            model_kwargs=self.cfg.model_kwargs,
            wavelet=self.cfg.wavelet,
            state_key=self.cfg.state_key,
            loader=self.cfg.loader,
            strict=self.cfg.strict,
            freeze=self.cfg.freeze,
            set_eval=self.cfg.set_eval,
        )
        self.model = components.model
        self.diffusion = components.diffusion
        self.idwt = components.idwt
        self._weights_loaded = True

    def sample(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
        ctx: InferenceContext | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return samples as (B,1,Z,Y,X) in [-1,1]."""
        if not self._weights_loaded:
            # In case someone instantiates without a runner that auto-loads wrapper weights.
            self.load_weights()

        assert self.model is not None and self.diffusion is not None and self.idwt is not None

        dev = device or next(self.model.parameters()).device

        model_dev = next(self.model.parameters()).device
        if model_dev != dev:
            self.model.to(dev)  # do not pass kwargs; upstream .to() override handles devices
        # Ensure the internal devices list exists (some forks keep it None unless .to() ran)
        if getattr(self.model, "devices", None) is None:
            p = next(self.model.parameters())
            try:
                self.model.devices = [p.device, p.device]
            except Exception:
                pass

        try:
            self.idwt.to(dev)
        except Exception:
            pass

        # dtype is ignored by default (keep model's dtype). Keep sampling noise in fp32.
        if self.cfg.seed is not None:
            torch.manual_seed(int(self.cfg.seed))
        # Wavelet-domain shape: (B, 8, image_size//2, image_size//2, image_size//2)
        d = int(self.cfg.image_size // self.cfg.wavelet_downsample)
        noise = torch.randn(
            (int(batch_size), int(self.cfg.wavelet_components), d, d, d),
            device=dev,
        )

        with torch.no_grad():
            sample = self.diffusion.p_sample_loop(
                model=self.model,
                shape=noise.shape,
                noise=noise,
                clip_denoised=bool(self.cfg.clip_denoised),
                model_kwargs={},  # no class/text conditioning in this checkpoint
                device=dev,
            )

            # IDWT expects 8 separate tensors with shape (B,1,D,H,W)
            B, C, D, H, W = sample.shape
            if C != int(self.cfg.wavelet_components):
                raise ValueError(f"Expected {self.cfg.wavelet_components} wavelet channels, got {C}")

            comps = [sample[:, i, :, :, :].view(B, 1, D, H, W) for i in range(C)]
            comps[0] = comps[0] * float(self.cfg.lll_scale)

            recon = self.idwt(*comps)  # (B,1,image_size,image_size,image_size)

            # keep in [-1,1] range; the generation pipeline can normalize to [0,1]
            recon = recon.clamp(-1.0, 1.0)

        return recon
