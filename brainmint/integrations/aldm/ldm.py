"""ALDM latent diffusion builders."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from torch import nn

from brainmint.integrations.aldm.repo import ALDM_LDM_CONFIG_RELPATH, ldm_import_context


def build_ldm_model(
    *,
    ldm_ckpt_path: str | Path,
    vqgan_stage2_ckpt_path: str | Path,
    ldm_config_path: str | Path | None = None,
) -> nn.Module:
    """Instantiate the full ALDM LDM model with checkpoint paths patched in."""

    ldm_ckpt = Path(ldm_ckpt_path).expanduser().resolve()
    if not ldm_ckpt.exists():
        raise FileNotFoundError(f"ALDM LDM checkpoint not found: {ldm_ckpt}")

    vqgan_stage2_ckpt = Path(vqgan_stage2_ckpt_path).expanduser().resolve()
    if not vqgan_stage2_ckpt.exists():
        raise FileNotFoundError(f"ALDM VQ-GAN stage-2 checkpoint not found: {vqgan_stage2_ckpt}")

    with ldm_import_context() as repo:
        from ldm.util import instantiate_from_config  # type: ignore

        cfg_path = repo.resolve_default_or_override(
            ldm_config_path,
            default_relpath=ALDM_LDM_CONFIG_RELPATH,
            label="ALDM LDM config",
        )
        base_cfg = OmegaConf.load(str(cfg_path))

        # Upstream constructors read these paths during instantiation.
        base_cfg.model.params.ckpt_path = str(ldm_ckpt)
        base_cfg.model.params.first_stage_config.params.ckpt_path = str(vqgan_stage2_ckpt)

        model = instantiate_from_config(base_cfg.model)
        model.eval()
        return model

