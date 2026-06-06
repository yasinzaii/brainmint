from __future__ import annotations

"""BrainMint-facing cWDM modality translation wrapper."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Tuple

import torch
from torch import nn

from brainmint.integrations.cwdm.translator import build_cwdm_model, build_cwdm_wavelet_layers
from brainmint.models.translation.utils import center_crop_or_pad_hwd, ensure_b1hwd, pick_first_tensor, resolve_device

CWDM_MODALITIES = ("t1w", "t2w", "flair", "t1ce")

CWDM_CONDITIONING_ORDER = {
    "t1w": ("t1ce", "t2w", "flair"),
    "t1ce": ("t1w", "t2w", "flair"),
    "t2w": ("t1w", "t1ce", "flair"),
    "flair": ("t1w", "t1ce", "t2w"),
}


@dataclass
class cWDMConfig:
    device: str | torch.device | None = "auto"
    cond_hwd: Tuple[int, int, int] = (224, 224, 160)

    ckpt_t1w: str = "MISSING"
    ckpt_t1ce: str = "MISSING"
    ckpt_t2w: str = "MISSING"
    ckpt_flair: str = "MISSING"

    clip_denoised: bool = True

    def checkpoint_map(self) -> dict[str, str]:
        return {
            "t1w": self.ckpt_t1w,
            "t1ce": self.ckpt_t1ce,
            "t2w": self.ckpt_t2w,
            "flair": self.ckpt_flair,
        }


def _to_config(cfg: cWDMConfig | Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> cWDMConfig:
    if cfg is None:
        data: dict[str, Any] = {}
    elif isinstance(cfg, cWDMConfig):
        data = asdict(cfg)
    elif isinstance(cfg, Mapping):
        data = dict(cfg)
    else:
        raise TypeError(f"cWDM cfg must be cWDMConfig/dict/None, got {type(cfg)!r}")

    data.update(dict(kwargs))
    return cWDMConfig(**data)


class cWDMModalityTranslator(nn.Module):
    """In-process cWDM translator exposed through BrainMint conventions."""

    def __init__(self, cfg: cWDMConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.cfg = _to_config(cfg, kwargs)
        self.device = resolve_device(self.cfg.device)
        self.dwt, self.idwt = build_cwdm_wavelet_layers()
        self.models = nn.ModuleDict()
        self.diffusion: Any | None = None

        for target, checkpoint in self.cfg.checkpoint_map().items():
            model, diffusion = build_cwdm_model(target=target, checkpoint=checkpoint)
            model.to(self.device)
            self.models[target] = model
            if self.diffusion is None:
                self.diffusion = diffusion

        if not self.models or self.diffusion is None:
            raise RuntimeError("No cWDM models were built. Check ckpt_t1w/ckpt_t2w/ckpt_flair/ckpt_t1ce paths.")

        self.dwt.to(self.device)
        self.idwt.to(self.device)
        self.eval()

    def _runtime_device(self) -> torch.device:
        try:
            return next(self.models.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("cWDM has no loaded target models.") from exc

    def _conditioning_for_target(self, batch: Mapping[str, object], target: str, reference: torch.Tensor) -> list[torch.Tensor]:
        conds: list[torch.Tensor] = []
        device = self._runtime_device()
        for modality in CWDM_CONDITIONING_ORDER[target]:
            value = batch.get(modality)
            tensor = ensure_b1hwd(value) if torch.is_tensor(value) else torch.zeros_like(reference)
            conds.append(center_crop_or_pad_hwd(tensor.to(device).float(), self.cfg.cond_hwd))
        return conds

    def _wavelet_conditioning(self, conds: list[torch.Tensor]) -> torch.Tensor:
        wavelet_conds: list[torch.Tensor] = []
        for cond in conds:
            lll, llh, lhl, lhh, hll, hlh, hhl, hhh = self.dwt(cond)
            lll = lll / 3.0
            wavelet_conds.append(torch.cat([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], dim=1))
        return torch.cat(wavelet_conds, dim=1)

    @torch.no_grad()
    def translate(self, *, batch: Mapping[str, object], target_modality: str) -> torch.Tensor:
        target = str(target_modality)
        if target not in CWDM_MODALITIES:
            raise ValueError(f"Unsupported target {target!r}. Expected one of {CWDM_MODALITIES}.")
        if target not in self.models:
            raise ValueError(f"Target {target!r} not available. Loaded: {sorted(self.models.keys())}")

        reference = pick_first_tensor(batch, ["t1w", "t2w", "flair", "t1ce"])
        if reference is None:
            raise KeyError("Batch contains no modality tensors; expected t1w/t2w/flair/t1ce")
        reference = ensure_b1hwd(reference)
        reference_hwd = (int(reference.shape[2]), int(reference.shape[3]), int(reference.shape[4]))

        device = self._runtime_device()
        conds = self._conditioning_for_target(batch, target, reference)
        cond_w = self._wavelet_conditioning(conds)
        batch_size, _, h2, w2, d2 = cond_w.shape
        noise = torch.randn((batch_size, 8, h2, w2, d2), device=device)

        sample = self.diffusion.p_sample_loop(
            self.models[target],
            noise.shape,
            noise=noise,
            cond=cond_w,
            device=device,
            clip_denoised=self.cfg.clip_denoised,
            progress=False,
        )

        sample = sample.float()
        sample[:, :1] = sample[:, :1] * 3.0
        reconstruction = self.idwt(
            sample[:, 0:1],
            sample[:, 1:2],
            sample[:, 2:3],
            sample[:, 3:4],
            sample[:, 4:5],
            sample[:, 5:6],
            sample[:, 6:7],
            sample[:, 7:8],
        )
        return center_crop_or_pad_hwd(reconstruction, reference_hwd).clamp(0.0, 1.0)

    @torch.no_grad()
    def forward(self, *, batch: Mapping[str, object], modality: str) -> torch.Tensor:  # type: ignore[override]
        return self.translate(batch=batch, target_modality=modality)


class cWDMMetricsGenerator(cWDMModalityTranslator):
    """Backward-compatible name used by existing metric/inference configs."""


__all__ = ["cWDMConfig", "cWDMModalityTranslator", "cWDMMetricsGenerator"]
