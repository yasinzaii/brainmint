from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.networks.schedulers import DDPMScheduler, RFlowScheduler

from brainmint.utils.state_dict_loader import load_weight_specs

_LOG = logging.getLogger(__name__)

@dataclass
class ControlNetInferenceOutputs:
    """Outputs for modality-translation inference."""
    sample: torch.Tensor                  # image-space (B, C, Z, Y, X)
    latent_mu: torch.Tensor               # unscaled latent mu (B, C_lat, Z, Y, X)
    latent_sigma: Optional[torch.Tensor]  # latent sigma from re-encoding (B, C_lat, Z, Y, X)


class ControlNetInferenceModule(nn.Module):
    """
    Inference-only ControlNet module for staged modality translation.

      - target_modality must be provided per call
      - stage-specific export configs bind exactly one controlnet_stage_* module
      - all configs provide hparams.control_keys_by_target
      - reference key is always T1w by default (hparams.reference_key)
    """

    # Fixed mapping used throughout the project.
    STAGE_TO_TARGET = {
        "stage_a": "t2w",
        "stage_b": "flair",
        "stage_c": "t1ce",
    }
    TARGET_TO_STAGE = {v: k for k, v in STAGE_TO_TARGET.items()}

    def __init__(
        self,
        autoencoder: Optional[nn.Module],
        diffusion_unet: nn.Module,
        noise_scheduler: Any,
        hparams: Dict[str, Any],
        # Stage A generates T2w
        controlnet_stage_a: Optional[nn.Module] = None,
        # Stage B generates FLAIR
        controlnet_stage_b: Optional[nn.Module] = None,
        # Stage C generates T1ce
        controlnet_stage_c: Optional[nn.Module] = None,
        demographics_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.hparams: Dict[str, Any] = dict(hparams) if hparams is not None else {}
        self._loaded_once = False
        self._loaded_targets: set[str] = set()

        # Modules
        self.autoencoder = autoencoder
        self.unet = diffusion_unet

        self.controlnet_stage_a = controlnet_stage_a
        self.controlnet_stage_b = controlnet_stage_b
        self.controlnet_stage_c = controlnet_stage_c

        self.demographics_encoder = demographics_encoder

        if noise_scheduler is None:
            raise ValueError("noise_scheduler must be provided")
        self.noise_scheduler = noise_scheduler

        hp = self.hparams

        # Required: scale factor
        _sf_val = hp.get("scale_factor", None)
        if _sf_val is None:
            raise ValueError("'scale_factor' must be provided in hparams (required for correct scaling).")
        sf = float(_sf_val)
        if sf <= 0:
            raise ValueError(f"'scale_factor' must be > 0, got {sf}.")
        self.register_buffer("_scale_factor", torch.tensor(sf, dtype=torch.float32), persistent=True)

        # Prediction type
        _pt = hp.get("prediction_type", getattr(self.noise_scheduler, "prediction_type", "v_prediction"))
        self.prediction_type = DDPMPredictionType(str(_pt))
        if hasattr(self.noise_scheduler, "prediction_type"):
            self.noise_scheduler.prediction_type = self.prediction_type

        # Basic configs
        self.num_inference_steps = int(hp.get("num_inference_steps", 50))
        self.conditioning_scale = float(hp.get("conditioning_scale", 1.0))

        # If True, refuse to run inference unless load_weights()/setup(stage=...) has run.
        # This prevents silent "all-noise" outputs.
        self.require_weights_loaded = bool(hp.get("require_weights_loaded", True))

        # Latent channels
        self.latent_channels = int(hp.get("latent_channels", getattr(self.unet, "in_channels", 4)))
        if getattr(self.unet, "in_channels", self.latent_channels) != self.latent_channels:
            raise ValueError(
                f"UNet in_channels={getattr(self.unet,'in_channels',None)} "
                f"!= hparams.latent_channels={self.latent_channels}"
            )

        # Reference key (always T1w by design; configurable but NOT per target).
        self.reference_key = str(hp.get("reference_key", "t1w")).strip().lower()
        if not self.reference_key:
            raise ValueError("hparams.reference_key must be a non-empty string (expected 't1w').")

        # Modality mapping / label range
        self.use_modality_mapping = bool(hp.get("use_modality_mapping", True))
        if not self.use_modality_mapping:
            raise ValueError("Modality translation inference requires `use_modality_mapping: true`.")

        self.modality_map = {str(k).lower(): int(v) for k, v in hp.get("modality_map", {}).items()}
        if not self.modality_map:
            raise ValueError("Missing `hparams.modality_map` for class-conditioned sampling.")
        vals = list(self.modality_map.values())
        self._modality_min = min(vals)
        self._modality_max = max(vals)

        # Control keys per generated target.
        self.control_keys_by_target: Dict[str, List[str]] = {
            str(k).strip().lower(): [str(x).strip().lower() for x in (v or [])]
            for k, v in dict(hp.get("control_keys_by_target", {}) or {}).items()
        }
        if not self.control_keys_by_target:
            raise ValueError("Inference requires non-empty hparams.control_keys_by_target")

        for required in ("t2w", "flair", "t1ce"):
            if required not in self.control_keys_by_target:
                raise ValueError(
                    f"hparams.control_keys_by_target missing required key '{required}'. "
                    f"Present keys: {sorted(self.control_keys_by_target.keys())}"
                )
            if not self.control_keys_by_target[required]:
                raise ValueError(f"hparams.control_keys_by_target['{required}'] is empty")

        self.control_concat_dim = int(hp.get("control_concat_dim", 1))
        self.control_apply_scale_factor = bool(hp.get("control_apply_scale_factor", True))

        # If True, batch[control_keys] are image-space tensors and we must encode them to latents.
        # If False, batch[control_keys] are already latents.
        self.control_in_image_space = bool(hp.get("control_in_image_space", False))

        # Synthetic embedding: batch MUST provide this key (if any stage needs it)
        self.is_mod_synthetic_key = str(hp.get("is_mod_synthetic_key", "")).strip() or None
        needs_syn = False
        for cn in [self.controlnet_stage_a, self.controlnet_stage_b, self.controlnet_stage_c]:
            if bool(getattr(cn, "use_synthetic_embedding", False)) and int(getattr(cn, "synthetic_cond_dim", 0) or 0) > 0:
                needs_syn = True
                break
        if needs_syn and not self.is_mod_synthetic_key:
            raise ValueError("Synthetic embedding is enabled but hparams.is_mod_synthetic_key is missing.")
        if self.is_mod_synthetic_key and not needs_syn:
            _LOG.warning(
                "hparams.is_mod_synthetic_key is set (%s) but none of the provided ControlNets require it. "
                "It will be ignored.",
                self.is_mod_synthetic_key,
            )

        # Optional sigma computation
        self.compute_output_sigma = bool(hp.get("compute_output_sigma", True))

        # Freeze all
        for m in [
            self.autoencoder,
            self.unet,
            self.controlnet_stage_a,
            self.controlnet_stage_b,
            self.controlnet_stage_c,
            self.demographics_encoder,
        ]:
            if m is None:
                continue
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    # ----------------------- helpers -----------------------

    def _is_rflow_scheduler(self) -> bool:
        inner = getattr(self.noise_scheduler, "_sched", None)
        return isinstance(self.noise_scheduler, RFlowScheduler) or isinstance(inner, RFlowScheduler)

    @property
    def device(self) -> torch.device:
        """Device used for inference tensors.

        This mirrors the useful part of LightningModule.device after converting
        the inference wrapper to a plain nn.Module: prefer registered parameter
        devices, then registered buffer devices.
        """

        for parameter in self.parameters(recurse=True):
            return parameter.device
        for buffer in self.buffers(recurse=True):
            return buffer.device
        return torch.device("cpu")

    def load_weights(self) -> None:
        if self._loaded_once:
            return

        specs = self.hparams.get("weight_loads", []) or []
        self._loaded_targets.update(load_weight_specs(self, specs))
        self._loaded_once = True

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        """Compatibility hook for scripts that previously used Lightning setup()."""

        self.load_weights()

    def _get_scale_factor(self, z: torch.Tensor) -> torch.Tensor:
        return self._scale_factor.to(device=z.device, dtype=z.dtype)

    def _validate_labels(self, labels: torch.Tensor) -> None:
        if labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("class_labels tensor must contain integers")
        if (labels < self._modality_min).any() or (labels > self._modality_max).any():
            raise ValueError(
                f"Invalid class labels {labels.tolist()}; valid range [{self._modality_min}, {self._modality_max}]"
            )

    def _resolve_controlnet(self, target_modality: str) -> nn.Module:
        stage = self.TARGET_TO_STAGE.get(target_modality.strip().lower())
        if stage is None:
            raise ValueError(
                f"Unsupported target_modality='{target_modality}'. Expected one of: {sorted(self.TARGET_TO_STAGE.keys())}"
            )

        cn = getattr(self, f"controlnet_{stage}", None)
        if cn is None:
            raise ValueError(
                f"Requested target '{target_modality}' requires {stage}, but that ControlNet was not provided. "
                f"(Provided: stage_a={self.controlnet_stage_a is not None}, "
                f"stage_b={self.controlnet_stage_b is not None}, "
                f"stage_c={self.controlnet_stage_c is not None})"
            )
        return cn

    def _control_keys_for(self, target_modality: str) -> List[str]:
        target = str(target_modality).strip().lower()
        keys = self.control_keys_by_target.get(target)
        if not keys:
            raise KeyError(
                f"Missing control_keys_by_target entry for target='{target}'. "
                f"Available: {sorted(self.control_keys_by_target.keys())}"
            )
        return list(keys)

    def _prep_class_labels(
        self,
        target_modality: str,
        batch_size: int,
    ) -> torch.Tensor:
        if not getattr(self.unet, "num_class_embeds", 0):
            raise KeyError("UNet is not class-conditioned but modality translation requires class conditioning.")

        target = str(target_modality).strip().lower()
        if target not in self.modality_map:
            raise KeyError(
                f"Unknown target modality '{target}'. modality_map keys={sorted(self.modality_map.keys())}"
            )
        label = int(self.modality_map[target])
        labels = torch.full((batch_size,), label, device=self.device, dtype=torch.long)
        self._validate_labels(labels)
        return labels

    def _encode_demographics(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        if not getattr(self.unet, "with_demographics", False):
            return None
        if self.demographics_encoder is None:
            raise RuntimeError("UNet expects demographics but demographics_encoder is None.")

        if "demo_values" not in batch or "demo_missing" not in batch:
            raise ValueError("Batch missing demo_values/demo_missing while UNet.with_demographics=True")

        demo_values = batch["demo_values"]
        demo_missing = batch["demo_missing"]
        if not torch.is_tensor(demo_values) or not torch.is_tensor(demo_missing):
            raise TypeError("demo_values and demo_missing must be tensors")
        if demo_values.shape != demo_missing.shape:
            raise ValueError(f"demo_values/demo_missing shape mismatch: {demo_values.shape} vs {demo_missing.shape}")

        demo_values = demo_values.to(self.device, non_blocking=True)
        demo_missing = demo_missing.to(self.device, non_blocking=True)

        with torch.no_grad():
            dem_emb = self.demographics_encoder(demo_values, demo_missing)

        if not torch.is_tensor(dem_emb):
            raise TypeError("demographics_encoder must return a torch.Tensor")
        if dem_emb.ndim != 2:
            raise ValueError(f"demographics embedding must be 2D (B,D), got {dem_emb.shape}")
        return dem_emb

    def _get_is_mod_synthetic(self, batch: Dict[str, Any], controlnet: nn.Module) -> Optional[torch.Tensor]:
        """If the selected controlnet uses synthetic embedding, batch MUST provide is_mod_synthetic."""
        if controlnet is None:
            return None

        if not bool(getattr(controlnet, "use_synthetic_embedding", False)):
            raise ValueError("All models trained with embeddings (For Now)")

        syn_dim = int(getattr(controlnet, "synthetic_cond_dim", 0) or 0)
        if syn_dim <= 0:
            raise ValueError("Synthetic Embedding Enabled but synthetic_cond_dim is invalid (<= 0)")

        if not self.is_mod_synthetic_key:
            raise KeyError("ControlNet requires synthetic embedding but hparams.is_mod_synthetic_key is not set.")
        if self.is_mod_synthetic_key not in batch:
            raise KeyError(f"Missing key '{self.is_mod_synthetic_key}' in batch (required for synthetic embedding).")

        v = batch[self.is_mod_synthetic_key]
        if not torch.is_tensor(v):
            raise TypeError(f"batch['{self.is_mod_synthetic_key}'] must be a torch.Tensor")
        v = v.to(self.device, non_blocking=True).float()

        if v.ndim != 2 or int(v.shape[1]) != syn_dim:
            raise ValueError(
                f"is_mod_synthetic must have shape (B, {syn_dim}) for this ControlNet, got {tuple(v.shape)}"
            )
        return v

    def _prep_controlnet_cond(self, batch: Dict[str, Any], control_keys: List[str], controlnet: nn.Module) -> torch.Tensor:
        xs: List[torch.Tensor] = []
        for k in control_keys:
            if k not in batch:
                raise KeyError(f"Missing control key {k!r} in batch")
            t = batch[k]
            if not torch.is_tensor(t):
                raise TypeError(f"Control key {k!r} must be a torch.Tensor")
            if t.ndim != 5:
                raise ValueError(f"Control tensor {k!r} must be 5D (B,C,Z,Y,X) got {t.shape}")
            t = t.to(self.device, non_blocking=True)

            if self.control_in_image_space:
                if self.autoencoder is None:
                    raise RuntimeError("control_in_image_space=True requires an autoencoder")
                t_mu, _ = self.encode_mu_sigma(t)
                t = t_mu

            if self.control_apply_scale_factor:
                t = t * self._get_scale_factor(t)

            xs.append(t)

        ctrl = torch.cat(xs, dim=self.control_concat_dim)

        expected_c = getattr(controlnet, "conditioning_embedding_in_channels", None)
        if expected_c is not None and int(ctrl.shape[1]) != int(expected_c):
            raise ValueError(f"ctrl C={int(ctrl.shape[1])} != ControlNet expected C={int(expected_c)}")

        return ctrl

    # ----------------------- encode/decode -----------------------

    @torch.no_grad()
    def decode_latent(self, z_unscaled: torch.Tensor) -> torch.Tensor:
        if self.autoencoder is None:
            raise ValueError("Autoencoder missing")
        dec = getattr(self.autoencoder, "decode", None)
        if callable(dec):
            return dec(z_unscaled)
        raise ValueError("Autoencoder missing callable decode()")

    @torch.no_grad()
    def encode_mu_sigma(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.autoencoder is None:
            raise RuntimeError("autoencoder is None but encode_mu_sigma was requested.")
        enc = getattr(self.autoencoder, "encode", None)
        with torch.autocast(device_type=self.device.type, enabled=False):
            if callable(enc):
                z_mu, z_sigma = enc(images.float())
            else:
                _, z_mu, z_sigma = self.autoencoder(images.float())
        return z_mu, z_sigma

    # ----------------------- sampling loop -----------------------

    @torch.no_grad()
    def sample_latent_controlled(
        self,
        output_size: Tuple[int, int, int],
        ctrl: torch.Tensor,
        controlnet: nn.Module,
        class_labels: torch.Tensor,
        demographics_embedding: Optional[torch.Tensor] = None,
        is_mod_synthetic: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        device = self.device
        b = ctrl.shape[0]
        z, y, x = output_size
        img = torch.randn((b, self.latent_channels, z, y, x), device=device)

        controlnet.eval()
        self.unet.eval()

        steps = int(num_inference_steps or self.num_inference_steps)
        if self._is_rflow_scheduler():
            self.noise_scheduler.set_timesteps(
                device=device,
                num_inference_steps=steps,
                input_img_size_numel=torch.tensor(output_size, device=device).prod().item(),
            )
        else:
            self.noise_scheduler.set_timesteps(num_inference_steps=steps)
            if isinstance(self.noise_scheduler, DDPMScheduler) and steps < self.noise_scheduler.num_train_timesteps:
                warnings.warn(
                    "DDPMScheduler num_inference_steps="
                    f"{steps} < num_train_timesteps={self.noise_scheduler.num_train_timesteps}"
                )

        all_t = self.noise_scheduler.timesteps.to(device)

        for i, t in enumerate(all_t):
            t_val = t.item()
            next_t_val = all_t[i + 1].item() if i + 1 < len(all_t) else 0
            t_batch = torch.full((b,), t_val, device=device, dtype=all_t.dtype)

            cn_kwargs: Dict[str, Any] = dict(
                x=img,
                timesteps=t_batch,
                controlnet_cond=ctrl,
                conditioning_scale=self.conditioning_scale,
                context=None,
                class_labels=class_labels,
            )
            if is_mod_synthetic is not None:
                cn_kwargs["is_mod_synthetic"] = is_mod_synthetic

            down_res, mid_res = controlnet(**cn_kwargs)

            unet_kwargs: Dict[str, Any] = dict(
                x=img,
                timesteps=t_batch,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
                class_labels=class_labels,
            )
            if demographics_embedding is not None:
                unet_kwargs["demographics_embedding"] = demographics_embedding

            pred = self.unet(**unet_kwargs)

            if self._is_rflow_scheduler():
                img, _ = self.noise_scheduler.step(pred, t_val, img, next_t_val)
            else:
                img, _ = self.noise_scheduler.step(pred, t_val, img)

        return img  # scaled latent

    # ----------------------- public wrappers -----------------------

    @torch.no_grad()
    def run_inference_bs(
        self,
        batch: Dict[str, Any],
        target_modality: str,
        num_inference_steps: Optional[int] = None,
    ) -> ControlNetInferenceOutputs:
        # Guard against running with random weights, a common cause of pure-noise outputs.
        if self.require_weights_loaded and not getattr(self, "_loaded_once", True):
            raise RuntimeError(
                "ControlNetInferenceModule weights have not been loaded yet. "
                "Call module.load_weights() or module.setup(stage='predict') before running inference."
            )

        # Reference is ALWAYS T1w (key = hparams.reference_key, default 't1w')
        if self.reference_key not in batch:
            raise KeyError(f"Missing reference_key={self.reference_key!r} from batch")

        ref = batch[self.reference_key]
        if not torch.is_tensor(ref) or ref.ndim != 5:
            raise ValueError(
                f"batch[{self.reference_key}] must be 5D tensor, got {type(ref)} {getattr(ref,'shape',None)}"
            )
        ref = ref.to(self.device, non_blocking=True)

        # Determine latent spatial size from reference (image-space requires VAE encode)
        if self.control_in_image_space:
            if self.autoencoder is None:
                raise RuntimeError("control_in_image_space=True requires an autoencoder")
            ref_mu, _ = self.encode_mu_sigma(ref)
            spatial = tuple(ref_mu.shape[-3:])
        else:
            if int(ref.shape[1]) != int(self.latent_channels):
                raise ValueError(
                    f"Reference latent C={int(ref.shape[1])} != expected latent_channels={self.latent_channels} "
                    f"(reference_key={self.reference_key!r})"
                )
            spatial = tuple(ref.shape[-3:])

        target = str(target_modality).strip().lower()
        controlnet = self._resolve_controlnet(target)
        control_keys = self._control_keys_for(target)

        ctrl = self._prep_controlnet_cond(batch=batch, control_keys=control_keys, controlnet=controlnet)
        labels = self._prep_class_labels(target_modality=target, batch_size=int(ctrl.shape[0]))
        dem_emb = self._encode_demographics(batch)
        is_syn = self._get_is_mod_synthetic(batch, controlnet)

        z_scaled = self.sample_latent_controlled(
            output_size=spatial,
            ctrl=ctrl,
            controlnet=controlnet,
            class_labels=labels,
            demographics_embedding=dem_emb,
            is_mod_synthetic=is_syn,
            num_inference_steps=num_inference_steps,
        )

        z_mu = z_scaled / self._get_scale_factor(z_scaled)
        sample = self.decode_latent(z_mu)

        z_sigma = None
        if self.compute_output_sigma:
            _, z_sigma = self.encode_mu_sigma(sample)

        return ControlNetInferenceOutputs(sample=sample, latent_mu=z_mu, latent_sigma=z_sigma)

    @torch.no_grad()
    def forward(
        self,
        batch: Dict[str, Any],
        modality: str,  # target modality
        num_inference_steps: Optional[int] = None,
    ) -> Any:
        """Call signature: module(batch, modality='t2w') -> generated image (B, C, Z, Y, X)."""
        out = self.run_inference_bs(
            batch=batch,
            target_modality=modality,
            num_inference_steps=num_inference_steps,
        )
        if self.control_in_image_space:
            return out.sample
        else:
            return out.sample, out.latent_mu


