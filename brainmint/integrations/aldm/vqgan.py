from __future__ import annotations

"""ALDM VQ-GAN builders.

This module is intentionally ALDM-specific. It owns upstream imports from
``taming`` and the stage-2 SPADE inference construction, while BrainMint-facing
compression/translation wrappers stay in ``brainmint.models``.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
from omegaconf import OmegaConf
from torch import nn

from brainmint.integrations.aldm.repo import (
    ALDM_VQGAN_STAGE1_CONFIG_RELPATH,
    ALDM_VQGAN_STAGE2_CONFIG_RELPATH,
    vqgan_import_context,
)

_LOG = logging.getLogger(__name__)


def build_stage1_vqgan(
    *,
    checkpoint_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> nn.Module:
    """Build ALDM's stage-1 VQ-GAN compression model for inference.

    The upstream config remains the source of truth for architecture. BrainMint
    patches only the checkpoint path and replaces the training loss with an
    identity module so reconstruction-only experiments do not instantiate
    discriminator/perceptual-loss components.
    """

    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ALDM VQ-GAN stage-1 checkpoint not found: {ckpt_path}")

    with vqgan_import_context() as repo:
        cfg_path = repo.resolve_default_or_override(
            config_path,
            default_relpath=ALDM_VQGAN_STAGE1_CONFIG_RELPATH,
            label="ALDM VQ-GAN stage-1 config",
        )

        from taming.models.vqgan import VQModel  # type: ignore

        base_cfg = OmegaConf.load(str(cfg_path))
        params = dict(OmegaConf.to_container(base_cfg.model.params, resolve=True))
        params["ckpt_path"] = str(ckpt_path)
        params["lossconfig"] = {"target": "torch.nn.Identity"}

        model = VQModel(**params)
        model.eval()
        return model


def build_stage2_spade_vqgan(
    *,
    checkpoint_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> nn.Module:
    """Build ALDM's stage-2 VQ-GAN/SPADE translator for inference.

    The upstream training config includes loss modules that can trigger network
    downloads. This inference-only builder constructs only the encoder,
    quantizer, decoder, and SPADE components required for translation.
    """

    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ALDM VQ-GAN stage-2 checkpoint not found: {ckpt_path}")

    with vqgan_import_context() as repo:
        cfg_path = repo.resolve_default_or_override(
            config_path,
            default_relpath=ALDM_VQGAN_STAGE2_CONFIG_RELPATH,
            label="ALDM VQ-GAN stage-2 config",
        )

        from taming.models.normalization import SPADEGenerator  # type: ignore
        from taming.modules.diffusionmodules.model import Decoder, Encoder  # type: ignore
        from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer  # type: ignore

        base_cfg = OmegaConf.load(str(cfg_path))
        params = base_cfg.model.params
        ddconfig = dict(OmegaConf.to_container(params.ddconfig, resolve=True))
        n_embed = int(params.n_embed)
        embed_dim = int(params.embed_dim)
        num_classes = int(params.num_classes)

        class VQGANStage2Inference(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = Encoder(**ddconfig)
                self.decoder = Decoder(**ddconfig)
                self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
                self.quant_conv = nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
                self.post_quant_conv = nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
                self.spade = SPADEGenerator(num_classes=num_classes, z_dim=ddconfig["z_channels"])

            def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
                h = self.encoder(x)
                h = self.quant_conv(h)
                quant, _, info = self.quantize(h)
                return quant, info

            def decode(self, quant: torch.Tensor) -> torch.Tensor:
                quant = self.post_quant_conv(quant)
                return self.decoder(quant)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                quant, _ = self.encode(x)
                quant = self.spade(quant, y)
                return self.decode(quant)

        model = VQGANStage2Inference()
        ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt_obj.get("state_dict", ckpt_obj)
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            _LOG.warning(
                "ALDM VQ-GAN stage-2 load_state_dict: %d missing keys, %d unexpected keys",
                len(missing),
                len(unexpected),
            )

        return model

