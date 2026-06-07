"""cWDM upstream modality-translation builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch import nn

from brainmint.integrations.cwdm.repo import cwdm_repo_context
from brainmint.utils.state_dict_loader import load_module_state_dict


@dataclass(frozen=True)
class CWDMUpstreamConfig:
    """Arguments expected by upstream cWDM ``create_model_and_diffusion``."""

    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str = ""
    image_size: int = 224
    num_channels: int = 64
    num_res_blocks: int = 2
    num_heads: int = 1
    num_heads_upsample: int = -1
    num_head_channels: int = -1
    attention_resolutions: str = ""
    channel_mult: str = "1,2,2,4,4"
    dropout: float = 0.0
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = False
    resblock_updown: bool = True
    use_fp16: bool = False
    use_new_attention_order: bool = False
    dims: int = 3
    num_groups: int = 32
    in_channels: int = 32  # 8 target wavelet channels + 24 conditioning wavelet channels.
    out_channels: int = 8
    bottleneck_attention: bool = False
    resample_2d: bool = False
    additive_skips: bool = False
    mode: str = "i2i"
    use_freq: bool = False
    predict_xstart: bool = True
    learn_sigma: bool = False
    class_cond: bool = False
    use_kl: bool = False
    rescale_timesteps: bool = False
    rescale_learned_sigmas: bool = False
    dataset: str = "brats"


def _load_cwdm_checkpoint(model: nn.Module, *, checkpoint_path: Path, target: str) -> None:
    """Load a cWDM root-level checkpoint into one target model."""

    load_module_state_dict(
        model,
        path=str(checkpoint_path),
        state_key="<root>",
        loader=None,
        strict=True,
        freeze=True,
        set_eval=True,
        target_name=f"cwdm_{target}",
    )


def build_cwdm_wavelet_layers(*, wavelet: str = "haar") -> tuple[nn.Module, nn.Module]:
    """Build the upstream DWT/IDWT helpers used by cWDM sampling."""

    with cwdm_repo_context():
        from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

        return DWT_3D(wavename=str(wavelet)), IDWT_3D(wavename=str(wavelet))


def build_cwdm_model(
    *,
    target: str,
    checkpoint: str | Path,
    config: CWDMUpstreamConfig | None = None,
) -> tuple[nn.Module, Any]:
    """Build one target-specific cWDM model and its diffusion object."""

    target = str(target)
    if checkpoint in (None, "", "MISSING"):
        raise ValueError(f"Missing cWDM checkpoint for target modality {target!r}")

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"cWDM checkpoint for {target!r} not found: {checkpoint_path}")

    create_kwargs = asdict(config or CWDMUpstreamConfig())

    with cwdm_repo_context():
        from guided_diffusion.script_util import create_model_and_diffusion

        model, diffusion = create_model_and_diffusion(**create_kwargs)
        diffusion.mode = "i2i"
        _load_cwdm_checkpoint(model, checkpoint_path=checkpoint_path, target=target)

    return model, diffusion
