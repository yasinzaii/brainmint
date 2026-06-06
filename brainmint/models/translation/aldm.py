from __future__ import annotations

"""BrainMint-facing ALDM modality translation wrapper.

There are two practical inference modes:

1. Full ALDM LDM sampling, using the upstream ``LDM/`` code and an LDM checkpoint.
2. VQ-GAN stage-2/SPADE translation only, using the upstream ``VQ-GAN/`` code and
   a stage-2 checkpoint.

This module owns BrainMint behavior: modality names, tensor shape conventions, and
input/output scaling. External ALDM imports and model construction live under
``brainmint.integrations.aldm``.
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from brainmint.integrations.aldm.ldm import build_ldm_model
from brainmint.integrations.aldm.vqgan import build_stage2_spade_vqgan
from brainmint.models.translation.utils import center_crop_or_pad_hwd, resolve_device

_LOG = logging.getLogger(__name__)


@dataclass
class ALDMConfig:
    # Which inference path to use.
    # - "ldm": full ALDM (LDM sampling + VQ-GAN stage-2 first stage)
    # - "vqgan_stage2": VQ-GAN stage-2 SPADE translation only
    mode: str = "ldm"

    # Optional upstream config overrides. None means use ALDM repo defaults.
    ldm_config_path: Optional[str] = None
    vqgan_stage2_config_path: Optional[str] = None

    # Checkpoints. Translation uses the LDM checkpoint and/or the VQ-GAN stage-2 checkpoint.
    ldm_ckpt_path: str = ""
    vqgan_stage2_ckpt_path: str = ""

    # ALDM modality label order (upstream BraTS): [t1, t1ce, t2, flair]
    modality_order: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair")

    # Sampling defaults for mode="ldm"
    ddim_steps: int = 200
    ddim_eta: float = 1.0

    # Device / precision
    device: str = "cuda"
    use_amp: bool = False


@dataclass
class ALDMTranslationGeneratorConfig:
    """Hydra-friendly config for using ALDM as an on-the-fly translation generator.

    - **Source is always T1w**.
      For metrics and inference, the conditioning batch is expected to provide `t1w`.
    - ALDM expects BraTS-shaped volumes **144x192x144 (H,W,D)**; we center crop/pad
      the input to this size and then center crop/pad the output back to the input
      spatial size so metrics are computed in your BrainScape ROI.
    - Inputs from your BrainScape transforms are in **[0,1]**; the ALDM wrapper
      converts internally to **[-1,1]** and returns **[0,1]**.
    """
    mode: str = "ldm"
    ldm_ckpt_path: str = ""
    vqgan_stage2_ckpt_path: str = ""
    ldm_config_path: Optional[str] = None
    vqgan_stage2_config_path: Optional[str] = None

    # ALDM/BraTS volume size in BrainMint tensor order (H,W,D).
    model_hwd: Sequence[int] = (144, 192, 144)

    ddim_steps: int = 50
    ddim_eta: float = 0.0
    device: str = "cuda"
    use_amp: bool = False

    # Safety clamp for outputs (after converting to [0,1])
    clamp_output: bool = True
    out_min: float = 0.0
    out_max: float = 1.0


def _norm_mod_name(name: str) -> str:
    n = str(name).strip().lower()
    if n in {"t1w", "t1"}:
        return "t1"
    if n in {"t1ce", "t1c", "t1gd", "t1c+", "t1cew"}:
        return "t1ce"
    if n in {"t2w", "t2"}:
        return "t2"
    if n in {"flair", "t2flair"}:
        return "flair"
    return n


class ALDMModalityTranslator(nn.Module):
    """In-process ALDM modality translator exposed through BrainMint conventions."""

    def __init__(self, cfg: Union[ALDMConfig, DictConfig, Mapping[str, Any]]) -> None:
        super().__init__()

        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, Mapping) and not isinstance(cfg, ALDMConfig):
            cfg = ALDMConfig(**dict(cfg))
        if not isinstance(cfg, ALDMConfig):
            raise TypeError(
                f"ALDMModalityTranslator cfg must be ALDMConfig/dict/DictConfig, got {type(cfg)}"
            )

        self.cfg: ALDMConfig = cfg
        self.model: Optional[nn.Module] = None
        self._weights_loaded: bool = False

    @staticmethod
    def _existing_path(path: str, *, label: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"{label} not found: {resolved}")
        return resolved

    def load_weights(self) -> None:
        if self._weights_loaded:
            return

        mode = str(self.cfg.mode).strip().lower()
        if mode not in {"ldm", "vqgan_stage2"}:
            raise ValueError(f"Unknown ALDM mode='{self.cfg.mode}'")

        if mode == "vqgan_stage2":
            if not self.cfg.vqgan_stage2_ckpt_path:
                raise ValueError("ALDMConfig.vqgan_stage2_ckpt_path is required for mode='vqgan_stage2'")

            stage2_ckpt = self._existing_path(
                self.cfg.vqgan_stage2_ckpt_path,
                label="ALDM VQ-GAN stage-2 checkpoint",
            )
            model = build_stage2_spade_vqgan(
                checkpoint_path=stage2_ckpt,
                config_path=self.cfg.vqgan_stage2_config_path,
            )
            dev = torch.device(self.cfg.device)
            model.to(dev)
            model.eval()
            self.model = model
            self._weights_loaded = True
            _LOG.info("Loaded ALDM VQ-GAN stage-2 (SPADE). device=%s", dev)
            return

        if not self.cfg.ldm_ckpt_path:
            raise ValueError("ALDMConfig.ldm_ckpt_path is required for mode='ldm'")
        if not self.cfg.vqgan_stage2_ckpt_path:
            raise ValueError("ALDMConfig.vqgan_stage2_ckpt_path is required for mode='ldm'")

        ldm_ckpt = self._existing_path(self.cfg.ldm_ckpt_path, label="ALDM LDM checkpoint")
        stage2_ckpt = self._existing_path(
            self.cfg.vqgan_stage2_ckpt_path,
            label="ALDM VQ-GAN stage-2 checkpoint",
        )
        model = build_ldm_model(
            ldm_config_path=self.cfg.ldm_config_path,
            ldm_ckpt_path=ldm_ckpt,
            vqgan_stage2_ckpt_path=stage2_ckpt,
        )
        dev = torch.device(self.cfg.device)
        model.to(dev)
        model.eval()
        self.model = model
        self._weights_loaded = True
        _LOG.info("Loaded ALDM (LDM + VQ-GAN). device=%s", dev)

    def _target_class(self, target_modality: str) -> int:
        m = _norm_mod_name(target_modality)
        order = tuple(_norm_mod_name(x) for x in self.cfg.modality_order)
        if m not in order:
            raise KeyError(f"Unknown target_modality='{target_modality}' (normalized='{m}'); allowed={order}")
        return int(order.index(m))

    @staticmethod
    def _ensure_b1hwd(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.dim() != 5 or x.shape[1] != 1:
            raise ValueError(f"ALDM expects source shaped (B,1,H,W,D) or (B,H,W,D); got {tuple(x.shape)}")
        return x

    @staticmethod
    def _to_m11(x01: torch.Tensor) -> torch.Tensor:
        return x01 * 2.0 - 1.0

    @staticmethod
    def _to_01(xm11: torch.Tensor) -> torch.Tensor:
        return (xm11 + 1.0) / 2.0

    @torch.no_grad()
    def translate(
        self,
        *,
        source: torch.Tensor,
        target_modality: str,
        ddim_steps: Optional[int] = None,
        ddim_eta: Optional[float] = None,
    ) -> torch.Tensor:
        return self.translate_one(
            source=source,
            target_modality=target_modality,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )

    @torch.no_grad()
    def translate_one(
        self,
        *,
        source: torch.Tensor,
        target_modality: str,
        ddim_steps: Optional[int] = None,
        ddim_eta: Optional[float] = None,
    ) -> torch.Tensor:
        """Translate a source volume to a target modality.

        Inputs are assumed to be in ``[0, 1]``. Outputs are returned in ``[0, 1]``.
        In ``vqgan_stage2`` mode, ``ddim_steps`` and ``ddim_eta`` are ignored.
        """

        if not self._weights_loaded:
            self.load_weights()
        assert self.model is not None

        model = self.model
        dev = next(model.parameters()).device

        x = self._ensure_b1hwd(source).to(device=dev, dtype=torch.float32)
        x = self._to_m11(x)

        y_cls = torch.full((x.shape[0],), self._target_class(target_modality), device=dev, dtype=torch.long)

        mode = str(self.cfg.mode).strip().lower()
        if mode == "vqgan_stage2":
            out = model(x, y_cls)
            out = out.clamp(-1.0, 1.0)
            return self._to_01(out).clamp(0.0, 1.0)

        steps = int(ddim_steps) if ddim_steps is not None else int(self.cfg.ddim_steps)
        eta = float(ddim_eta) if ddim_eta is not None else float(self.cfg.ddim_eta)

        ema_ctx = model.ema_scope("aldm_sampling") if hasattr(model, "ema_scope") else nullcontext()
        amp = bool(self.cfg.use_amp)

        z_src = model.encode_first_stage(x)
        z_srctgt = model.first_stage_model.spade(z_src, y_cls)
        z_src = model.get_first_stage_encoding(z_src)
        z_srctgt = model.get_first_stage_encoding(z_srctgt)
        z_cond = torch.cat([z_src, z_srctgt], dim=1).detach()
        cond = {"c_concat": z_cond, "c_crossattn": y_cls}

        with ema_ctx:
            if amp and dev.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    z, _ = model.sample_log(cond=cond, batch_size=int(x.shape[0]), ddim=True, ddim_steps=steps, eta=eta)
            else:
                z, _ = model.sample_log(cond=cond, batch_size=int(x.shape[0]), ddim=True, ddim_steps=steps, eta=eta)

        out = model.decode_first_stage(z)
        out = out.clamp(-1.0, 1.0)
        return self._to_01(out).clamp(0.0, 1.0)


class ALDMTranslationGenerator:
    """Batch-callable ALDM translator used by metrics and inference configs.

    Contract:
        ``generator(batch=batch, modality="t2w") -> (B,1,H,W,D)`` in ``[0,1]``.

    Source is always T1w, provided under ``batch["t1w"]``. Inputs are center
    cropped/padded to the ALDM model shape before translation, then restored to
    the original input spatial shape for metric computation.
    """

    def __init__(self, **cfg: Any) -> None:
        self.cfg = ALDMTranslationGeneratorConfig(**cfg)
        self._device = resolve_device(self.cfg.device)

        mode = str(self.cfg.mode).strip().lower()
        if mode == "vqgan_stage2":
            if not self.cfg.vqgan_stage2_ckpt_path:
                raise ValueError("ALDMTranslationGenerator requires vqgan_stage2_ckpt_path for mode='vqgan_stage2'")
        elif mode == "ldm":
            if not self.cfg.ldm_ckpt_path:
                raise ValueError("ALDMTranslationGenerator requires ldm_ckpt_path for mode='ldm'")
            if not self.cfg.vqgan_stage2_ckpt_path:
                raise ValueError("ALDMTranslationGenerator requires vqgan_stage2_ckpt_path for mode='ldm'")
        else:
            raise ValueError(f"Unknown ALDM mode={self.cfg.mode!r}")

        translator_cfg = ALDMConfig(
            mode=self.cfg.mode,
            ldm_config_path=self.cfg.ldm_config_path,
            vqgan_stage2_config_path=self.cfg.vqgan_stage2_config_path,
            ldm_ckpt_path=self.cfg.ldm_ckpt_path,
            vqgan_stage2_ckpt_path=self.cfg.vqgan_stage2_ckpt_path,
            ddim_steps=self.cfg.ddim_steps,
            ddim_eta=self.cfg.ddim_eta,
            device=str(self._device),
            use_amp=self.cfg.use_amp,
        )
        self._translator = ALDMModalityTranslator(translator_cfg)

    @torch.no_grad()
    def __call__(self, *, batch: Mapping[str, Any], modality: str) -> torch.Tensor:
        source = batch.get("t1w")
        if not torch.is_tensor(source):
            raise KeyError("ALDM generator expects a T1w tensor in batch['t1w']")

        source = ALDMModalityTranslator._ensure_b1hwd(source)
        input_hwd = tuple(int(value) for value in source.shape[-3:])
        model_hwd = tuple(int(value) for value in self.cfg.model_hwd)

        source_model = center_crop_or_pad_hwd(source, model_hwd)
        translated = self._translator.translate_one(
            source=source_model,
            target_modality=str(modality),
            ddim_steps=self.cfg.ddim_steps,
            ddim_eta=self.cfg.ddim_eta,
        )
        output = center_crop_or_pad_hwd(translated, input_hwd)

        if self.cfg.clamp_output:
            output = output.clamp(float(self.cfg.out_min), float(self.cfg.out_max))
        return output


__all__ = [
    "ALDMConfig",
    "ALDMModalityTranslator",
    "ALDMTranslationGenerator",
    "ALDMTranslationGeneratorConfig",
]
