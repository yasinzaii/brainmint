from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf  # type: ignore
from torch import nn

from brainmint.integrations.maisi.generator import MAISISampler, build_maisi_sampler

log = logging.getLogger(__name__)


@dataclass
class MAISI3DWrapperConfig:
    """BrainMint-facing config for MAISI / NV-Generate-MR generation."""

    # Network definitions as Hydra _target_ mappings.
    autoencoder_def: Mapping[str, Any] | None = None
    diffusion_unet_def: Mapping[str, Any] | None = None
    noise_scheduler_def: Mapping[str, Any] | None = None

    # Checkpoints
    autoencoder_ckpt_path: str = ""
    diffusion_ckpt_path: str = ""
    autoencoder_state_key: str | None = "unet_state_dict"
    diffusion_state_key: str | None = "unet_state_dict"
    strict: bool | str = True
    freeze: bool = True
    set_eval: bool = True

    # Sampling settings
    num_inference_steps: int = 1000
    latent_channels: int | None = None
    latent_divisor: int = 4

    # Output size / spacing (ZYX order to match GenMriStudy generation pipelines)
    output_size_zyx: Sequence[int] = (256, 256, 256)
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0)

    # Constant conditioning (MONAI MAISI style)
    top_region_index: Sequence[float] = (1.0, 0.0, 0.0, 0.0)
    bottom_region_index: Sequence[float] = (1.0, 0.0, 0.0, 0.0)
    modality: str | int = "mri_t1"
    modality_mapping_override: dict[str, int] | None = None
    body_region: Sequence[str] = ("head",)
    anatomy_list: Sequence[str] = ("brain",)
    label_dict_json: str | None = None

    # Scale factor fallback when the diffusion checkpoint does not store one.
    scale_factor: float | None = None

    # Optional clamp if downstream expects unit-range output.
    clamp_unit_range: bool = False

    # Autocast during sampling
    autocast: bool = True

    # Optional deterministic seed
    seed: int | None = None


def _to_config(cfg: Any) -> Any:
    """Keep Hydra compatibility while preserving instantiated module configs."""

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, Mapping):
        return MAISI3DWrapperConfig(**dict(cfg))
    return cfg


class MAISI3DWrapper(nn.Module):
    """Thin BrainMint-facing wrapper around a loaded MAISI integration sampler."""

    def __init__(self, *, cfg: MAISI3DWrapperConfig | DictConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = _to_config(cfg)
        self._device_ref = nn.Parameter(torch.empty(0), requires_grad=False)
        self.sampler: MAISISampler | None = None

        # Compatibility attributes for callers that inspect loaded modules.
        self.autoencoder: nn.Module | None = None
        self.unet: nn.Module | None = None
        self.noise_scheduler: Any | None = None
        self.scale_factor: torch.Tensor | None = None
        self._loaded = False

    def load_weights(self) -> None:
        """Build the loaded MAISI sampler through the integration layer."""

        if self._loaded:
            return

        sampler = build_maisi_sampler(self.cfg, device=self._device_ref.device)
        self.sampler = sampler
        self.autoencoder = sampler.autoencoder
        self.unet = sampler.unet
        self.noise_scheduler = sampler.noise_scheduler
        self.scale_factor = sampler.scale_factor
        self._loaded = True

        log.info(
            "Loaded MAISI sampler. output_size_zyx=%s spacing_zyx=%s modality=%s sf=%.6f",
            list(sampler.output_size_zyx),
            list(sampler.spacing_zyx),
            getattr(self.cfg, "modality", "mri_t1"),
            float(sampler.scale_factor.item()),
        )

    @torch.no_grad()
    def sample(
        self,
        batch: Mapping[str, Any] | None = None,
        *,
        batch_size: int,
        ctx: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Generate a batch of synthetic volumes. Returns ``(B,1,Z,Y,X)``."""

        if not self._loaded:
            self.load_weights()
        if self.sampler is None:
            raise RuntimeError("MAISI sampler was not loaded.")

        dev = device or self._device_ref.device
        return self.sampler.sample(batch, batch_size=batch_size, device=dev, dtype=dtype)


__all__ = ["MAISI3DWrapper", "MAISI3DWrapperConfig"]
