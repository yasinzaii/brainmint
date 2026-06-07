"""MAISI / NV-Generate-MR generator integration."""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from brainmint.integrations.maisi.repo import maisi_repo_context
from brainmint.utils.state_dict_loader import load_module_state_dict

_LOG = logging.getLogger(__name__)


def normalize_modality(modality: str | int, mapping: Mapping[str, int]) -> int:
    """Map common BrainMint modality strings to MAISI integer IDs."""

    if isinstance(modality, int):
        return int(modality)

    key = str(modality).strip().lower()
    key = {
        "t1w": "mri_t1",
        "t2w": "mri_t2",
        "flair": "mri_flair",
        "t1ce": "mri_t1ce",
    }.get(key, key)

    if key not in mapping:
        raise KeyError(
            f"Unknown MAISI modality {modality!r}. Known: {sorted(mapping.keys())} "
            "(from NV-Generate-MR configs/modality_mapping.json)."
        )
    return int(mapping[key])


def load_modality_mapping(repo_root: Path, overrides: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Load NV-Generate-MR modality IDs and apply optional BrainMint overrides."""

    raw = json.loads((repo_root / "configs" / "modality_mapping.json").read_text(encoding="utf-8"))
    mapping = {str(key).lower(): int(value) for key, value in raw.items()}
    if overrides:
        mapping.update({str(key).lower(): int(value) for key, value in overrides.items()})
    if not mapping:
        raise RuntimeError(f"MAISI modality mapping is empty under {repo_root / 'configs' / 'modality_mapping.json'}")
    return mapping


@dataclass(frozen=True)
class MAISIGenerationComponents:
    autoencoder: nn.Module
    unet: nn.Module
    noise_scheduler: Any
    scale_factor: torch.Tensor
    rflow_scheduler_cls: type
    set_determinism: Any
    modality_mapping: dict[str, int]


def _checkpoint_scale_factor(ckpt_path: str) -> Any:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, Mapping):
        return checkpoint.get("scale_factor")
    return None


def _resolve_scale_factor(*, cfg_scale_factor: Any, checkpoint_scale_factor: Any, device: torch.device) -> tuple[torch.Tensor, str]:
    source = "checkpoint" if checkpoint_scale_factor is not None else "config"
    value = checkpoint_scale_factor if checkpoint_scale_factor is not None else cfg_scale_factor
    if value is None:
        raise ValueError(
            "MAISI scale_factor is required. Use a checkpoint that contains 'scale_factor', or provide cfg.scale_factor."
        )
    value = float(value.item()) if torch.is_tensor(value) else float(value)
    return torch.tensor(value, device=device, dtype=torch.float32), source


class MAISISampler(nn.Module):
    """Loaded NV-Generate-MR/MAISI sampler with upstream-specific sampling behavior."""

    def __init__(self, *, cfg: Any, components: MAISIGenerationComponents) -> None:
        super().__init__()
        self.cfg = cfg
        self.autoencoder = components.autoencoder
        self.unet = components.unet
        self.noise_scheduler = components.noise_scheduler
        self.rflow_scheduler_cls = components.rflow_scheduler_cls
        self.set_determinism = components.set_determinism
        self.modality_mapping = components.modality_mapping
        self.register_buffer("scale_factor", components.scale_factor.detach().clone().float())

    @property
    def output_size_zyx(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in getattr(self.cfg, "output_size_zyx", (256, 256, 256)))

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in getattr(self.cfg, "spacing_zyx", (1.0, 1.0, 1.0)))

    def _runtime_device(self, device: torch.device | None = None) -> torch.device:
        if device is not None:
            return torch.device(device)
        try:
            return next(self.unet.parameters()).device
        except StopIteration:
            return self.scale_factor.device

    @torch.no_grad()
    def sample(
        self,
        batch: Mapping[str, Any] | None = None,
        *,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Generate synthetic volumes as ``(B, 1, Z, Y, X)``."""

        dev = self._runtime_device(device)
        if next(self.unet.parameters()).device != dev:
            self.to(dev)

        out_size = self.output_size_zyx
        spacing = self.spacing_zyx

        seed = getattr(self.cfg, "seed", None)
        if seed is not None:
            self.set_determinism(int(seed))

        bsz = int(batch_size)
        include_body_region = bool(getattr(self.unet, "include_top_region_index_input", False))
        include_modality = getattr(self.unet, "num_class_embeds", None) is not None

        spacing_tensor = torch.tensor(spacing, device=dev, dtype=torch.float16).unsqueeze(0)
        spacing_tensor = (spacing_tensor * 1e2).repeat(bsz, 1)

        top_region = getattr(self.cfg, "top_region_index", (1.0, 0.0, 0.0, 0.0))
        bottom_region = getattr(self.cfg, "bottom_region_index", (1.0, 0.0, 0.0, 0.0))
        top_tensor = torch.tensor(top_region, device=dev, dtype=torch.float16).unsqueeze(0)
        bottom_tensor = torch.tensor(bottom_region, device=dev, dtype=torch.float16).unsqueeze(0)
        top_tensor = (top_tensor * 1e2).repeat(bsz, 1)
        bottom_tensor = (bottom_tensor * 1e2).repeat(bsz, 1)

        if include_modality:
            modality_id = normalize_modality(getattr(self.cfg, "modality", "mri_t1"), self.modality_mapping)
            modality_tensor = torch.full((bsz,), modality_id, device=dev, dtype=torch.long)
        else:
            modality_tensor = None

        latent_channels = int(getattr(self.cfg, "latent_channels", None) or getattr(self.unet, "in_channels", 4))
        latent_divisor = int(getattr(self.cfg, "latent_divisor", 4))
        z, y, x = out_size
        image = torch.randn(
            (bsz, latent_channels, z // latent_divisor, y // latent_divisor, x // latent_divisor),
            device=dev,
        )

        input_numel = int(torch.prod(torch.tensor(image.shape[2:], device=dev)).item())
        num_inference_steps = int(getattr(self.cfg, "num_inference_steps", 1000))
        try:
            self.noise_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                input_img_size_numel=input_numel,
            )
        except TypeError:
            self.noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        self.autoencoder.eval()
        self.unet.eval()
        timesteps = self.noise_scheduler.timesteps
        next_timesteps = torch.cat(
            (timesteps[1:], torch.tensor([0], dtype=timesteps.dtype, device=timesteps.device))
        )
        inner_scheduler = getattr(self.noise_scheduler, "_sched", None)
        is_rflow = isinstance(self.noise_scheduler, self.rflow_scheduler_cls) or isinstance(inner_scheduler, self.rflow_scheduler_cls)
        use_autocast = bool(getattr(self.cfg, "autocast", True) and dev.type == "cuda")

        with torch.amp.autocast("cuda" if dev.type == "cuda" else "cpu", enabled=use_autocast):
            for t, next_t in zip(timesteps, next_timesteps, strict=True):
                unet_inputs: dict[str, Any] = {
                    "x": image,
                    "timesteps": torch.tensor(
                        (float(t.item() if torch.is_tensor(t) else t),), device=dev, dtype=torch.float32
                    ),
                    "spacing_tensor": spacing_tensor,
                }
                if include_body_region:
                    unet_inputs.update(
                        {
                            "top_region_index_tensor": top_tensor,
                            "bottom_region_index_tensor": bottom_tensor,
                        }
                    )
                if include_modality and modality_tensor is not None:
                    unet_inputs["class_labels"] = modality_tensor

                model_output = self.unet(**unet_inputs)
                if is_rflow:
                    image, _ = self.noise_scheduler.step(model_output, t, image, next_t)  # type: ignore[misc]
                else:
                    image, _ = self.noise_scheduler.step(model_output, t, image)  # type: ignore[misc]

        reconstruction = self.autoencoder.decode_stage_2_outputs(image / self.scale_factor.to(device=dev))
        if getattr(self.cfg, "clamp_unit_range", False):
            reconstruction = reconstruction.clamp(0.0, 1.0)
        if dtype is not None:
            reconstruction = reconstruction.to(dtype=dtype)
        return reconstruction


def build_maisi_sampler(cfg: Any, *, device: torch.device) -> MAISISampler:
    """Build a loaded MAISI sampler from a BrainMint-facing config object."""

    autoencoder_def = getattr(cfg, "autoencoder_def", None)
    diffusion_unet_def = getattr(cfg, "diffusion_unet_def", None)
    noise_scheduler_def = getattr(cfg, "noise_scheduler_def", None)
    if autoencoder_def is None or diffusion_unet_def is None or noise_scheduler_def is None:
        raise ValueError(
            "MAISI3DWrapper requires autoencoder_def, diffusion_unet_def, and noise_scheduler_def "
            "Hydra _target_ mappings."
        )

    with maisi_repo_context() as repo_root:
        from monai.networks.schedulers import RFlowScheduler
        from monai.utils import set_determinism
        from scripts.utils import define_instance  # type: ignore

        mapping = load_modality_mapping(repo_root, overrides=getattr(cfg, "modality_mapping_override", None))
        label_dict_json = getattr(cfg, "label_dict_json", None)
        label_path = Path(label_dict_json) if label_dict_json else (repo_root / "configs" / "label_dict.json")
        if not label_path.exists():
            raise FileNotFoundError(
                f"MAISI label_dict_json not found at {label_path}. Set cfg.label_dict_json to a valid JSON path."
            )

        if not isinstance(autoencoder_def, Mapping):
            raise TypeError("autoencoder_def must be a Hydra _target_ mapping.")
        if not isinstance(diffusion_unet_def, Mapping):
            raise TypeError("diffusion_unet_def must be a Hydra _target_ mapping.")
        if not isinstance(noise_scheduler_def, Mapping):
            raise TypeError("noise_scheduler_def must be a Hydra _target_ mapping.")

        autoencoder = define_instance(
            Namespace(autoencoder_def=dict(autoencoder_def)),
            "autoencoder_def",
        ).to(device)
        unet = define_instance(
            Namespace(diffusion_unet_def=dict(diffusion_unet_def)),
            "diffusion_unet_def",
        ).to(device)
        noise_scheduler = define_instance(Namespace(noise_scheduler=dict(noise_scheduler_def)), "noise_scheduler")

        strict = getattr(cfg, "strict", True)
        freeze = bool(getattr(cfg, "freeze", True))
        set_eval = bool(getattr(cfg, "set_eval", True))
        autoencoder_ckpt_path = str(cfg.autoencoder_ckpt_path)
        diffusion_ckpt_path = str(cfg.diffusion_ckpt_path)

        load_module_state_dict(
            autoencoder,
            path=autoencoder_ckpt_path,
            state_key=getattr(cfg, "autoencoder_state_key", "<root>"),
            strict=strict,
            freeze=freeze,
            set_eval=set_eval,
            target_name="maisi_autoencoder",
        )
        load_module_state_dict(
            unet,
            path=diffusion_ckpt_path,
            state_key=getattr(cfg, "diffusion_state_key", "unet_state_dict"),
            strict=strict,
            freeze=freeze,
            set_eval=set_eval,
            target_name="maisi_diffusion_unet",
        )
        cfg_scale_factor = getattr(cfg, "scale_factor", None)
        checkpoint_scale_factor = _checkpoint_scale_factor(diffusion_ckpt_path)
        scale_factor, scale_source = _resolve_scale_factor(
            cfg_scale_factor=cfg_scale_factor,
            checkpoint_scale_factor=checkpoint_scale_factor,
            device=device,
        )

        components = MAISIGenerationComponents(
            autoencoder=autoencoder,
            unet=unet,
            noise_scheduler=noise_scheduler,
            scale_factor=scale_factor,
            rflow_scheduler_cls=RFlowScheduler,
            set_determinism=set_determinism,
            modality_mapping=mapping,
        )
        _LOG.info("Built MAISI sampler with scale_factor source=%s", scale_source)
        return MAISISampler(cfg=cfg, components=components)
