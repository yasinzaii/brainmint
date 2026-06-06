from __future__ import annotations

"""Med-DDPM upstream generator builders."""

from pathlib import Path
from typing import Optional

from torch import nn

from brainmint.integrations.med_ddpm.repo import med_ddpm_repo_context
from brainmint.utils.state_dict_loader import load_module_state_dict


def build_med_ddpm_diffusion(
    *,
    ckpt_path: str | Path,
    image_size: int = 192,
    depth_size: int = 144,
    num_channels: int = 64,
    num_res_blocks: int = 2,
    timesteps: int = 250,
    with_condition: bool = True,
    out_channels: int = 4,
    state_key: Optional[str] = "ema",
    loader: Optional[str] = None,
    strict: bool | str = True,
    freeze: bool = True,
    set_eval: bool = True,
) -> nn.Module:
    """Build the upstream Med-DDPM diffusion module and load checkpoint weights."""

    checkpoint = Path(ckpt_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Med-DDPM checkpoint not found: {checkpoint}")

    with med_ddpm_repo_context():
        from diffusion_model.trainer_brats import GaussianDiffusion
        from diffusion_model.unet_brats import create_model

        model = create_model(
            image_size=int(image_size),
            num_channels=int(num_channels),
            num_res_blocks=int(num_res_blocks),
            in_channels=int(out_channels) * 2,
            out_channels=int(out_channels),
        )

        diffusion = GaussianDiffusion(
            model,
            image_size=int(image_size),
            depth_size=int(depth_size),
            channels=int(out_channels),
            timesteps=int(timesteps),
            loss_type="l1",
            with_condition=bool(with_condition),
        )

    load_module_state_dict(
        diffusion,
        path=str(checkpoint),
        state_key=state_key,
        loader=loader,
        strict=strict,
        freeze=freeze,
        set_eval=set_eval,
        target_name="med_ddpm",
    )
    return diffusion
