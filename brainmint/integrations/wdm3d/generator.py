from __future__ import annotations

"""WDM-3D upstream generator builders."""

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from torch import nn

from brainmint.integrations.wdm3d.repo import wdm3d_repo_context
from brainmint.utils.state_dict_loader import load_module_state_dict

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class WDM3DComponents:
    model: nn.Module
    diffusion: Any
    idwt: nn.Module


def wdm3d_preset_overrides(preset: str) -> dict[str, Any]:
    """Return create_model_and_diffusion overrides for known WDM-3D checkpoints."""

    preset = str(preset)
    common = dict(
        image_size=128,
        diffusion_steps=1000,
        noise_schedule="linear",
        num_channels=64,
        num_res_blocks=2,
        num_heads=1,
        num_groups=32,
        attention_resolutions="",
        channel_mult="1,2,2,4,4",
        dropout=0.0,
        class_cond=False,
        learn_sigma=False,
        use_scale_shift_norm=False,
        use_fp16=False,
        use_new_attention_order=False,
        bottleneck_attention=False,
        resample_2d=False,
        additive_skips=True,
        in_channels=8,
        out_channels=8,
        dims=3,
        mode="default",
        use_freq=False,
        dataset="brats",
        predict_xstart=True,
        renormalize=True,
    )

    if preset in ("brats_unet_128", "ours_unet_128"):
        return dict(common)
    if preset == "ours_unet_256":
        values = dict(common)
        values.update(image_size=256, channel_mult="1,1,2,2,4,4,4")
        return values
    if preset == "ours_wnet_128":
        values = dict(common)
        values.update(use_freq=True)
        return values
    if preset == "ours_wnet_256":
        values = dict(common)
        values.update(image_size=256, channel_mult="1,1,2,2,4,4,4", use_freq=True)
        return values

    raise ValueError(
        f"Unknown WDM preset {preset!r}. Known: brats_unet_128, "
        "ours_unet_128/256, ours_wnet_128/256"
    )


def build_wdm3d_components(
    *,
    ckpt_path: str | Path,
    preset: str = "brats_unet_128",
    model_kwargs: Optional[dict[str, Any]] = None,
    wavelet: str = "haar",
    state_key: Optional[str] = "<root>",
    loader: Optional[str] = None,
    strict: bool | str = True,
    freeze: bool = True,
    set_eval: bool = True,
) -> WDM3DComponents:
    """Build WDM-3D model, diffusion object, IDWT layer, and load weights."""

    checkpoint = Path(ckpt_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"WDM3D checkpoint not found: {checkpoint}")

    with wdm3d_repo_context():
        from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

        try:
            from DWT_IDWT.DWT_IDWT_layer import IDWT_3D
        except ModuleNotFoundError:
            from utils.dwt_utils import IDWT_3D

        args = dict(model_and_diffusion_defaults())
        args.update(wdm3d_preset_overrides(preset))
        if model_kwargs:
            args.update(dict(model_kwargs))

        allowed = set(inspect.signature(create_model_and_diffusion).parameters)
        filtered_args = {key: value for key, value in args.items() if key in allowed}
        dropped = sorted(set(args) - set(filtered_args))
        if dropped:
            _LOG.info("WDM3D dropping non-create_model_and_diffusion kwargs: %s", dropped)

        _LOG.info(
            "WDM3D create_model_and_diffusion args: %s",
            {
                "image_size": filtered_args.get("image_size"),
                "num_channels": filtered_args.get("num_channels"),
                "num_res_blocks": filtered_args.get("num_res_blocks"),
                "channel_mult": filtered_args.get("channel_mult"),
                "attention_resolutions": filtered_args.get("attention_resolutions"),
                "num_heads": filtered_args.get("num_heads"),
                "in_channels": filtered_args.get("in_channels"),
                "out_channels": filtered_args.get("out_channels"),
                "additive_skips": filtered_args.get("additive_skips"),
                "use_freq": filtered_args.get("use_freq"),
                "mode": filtered_args.get("mode"),
                "dataset": filtered_args.get("dataset"),
            },
        )

        model, diffusion = create_model_and_diffusion(**filtered_args)
        load_module_state_dict(
            model,
            path=str(checkpoint),
            state_key=state_key,
            loader=loader,
            strict=strict,
            freeze=freeze,
            set_eval=set_eval,
            target_name="wdm3d",
        )
        idwt = IDWT_3D(str(wavelet))

    return WDM3DComponents(model=model, diffusion=diffusion, idwt=idwt)
