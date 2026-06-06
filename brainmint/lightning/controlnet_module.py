"""PyTorch Lightning module for training a ControlNet on top of a frozen UNet.

This module started life as a BrainScape tumour-mask ControlNet trainer.
It has been generalized so the **same LightningModule** can be used for:

* tumour mask + masked-image guidance (image-space conditioning), and
* latent-space modality translation (e.g. T1w→T2w), or any other control signal.

Key design points
-----------------
* The **Diffusion UNet is frozen**; only the ControlNet is trained.
* The control input is built by concatenating a configurable list of batch keys
  (``hparams.control_keys``). If omitted, we fall back to the legacy
  ``image_masked`` + ``mask_one_hot`` behavior.
* When the control input is already in latent space, you should enable
  ``hparams.control_apply_scale_factor`` to match the scaled-latent convention.
* Optional demographics conditioning is supported **only for the UNet**
  (ControlNet stays generic and does not take demographics).
* Optional non-spatial conditioning vector ``is_mod_synthetic`` can be passed
  to the ControlNet (if the ControlNet was configured with
  ``use_synthetic_embedding: true``).
"""
import contextlib
import functools
import logging
import warnings
from typing import Any, Dict, Optional, Sequence, Union, Tuple, List

import torch
import torch.nn as nn
import pytorch_lightning as pl

from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.networks.schedulers import RFlowScheduler, DDPMScheduler

from brainmint.utils.gpumem_utils import SimpleGPUMemoryTracker
from brainmint.utils.state_dict_loader import StateDictLoaderMixin
from brainmint.utils.ema import EMAConfig, ExponentialMovingAverage

from omegaconf import OmegaConf

_LOG = logging.getLogger(__name__)

def _to_plain(x):
    # Convert OmegaConf containers -> plain Python recursively
    if OmegaConf.is_config(x):
        return OmegaConf.to_container(x, resolve=True, throw_on_missing=False)
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_plain(v) for v in x]
    return x

class ControlNetModule(StateDictLoaderMixin, pl.LightningModule):
    """
    PyTorch Lightning module for training ControlNet on top of a frozen
    diffusion UNet.  Only the ControlNet is updated during training.

    Args:
        autoencoder: Optional autoencoder used to encode images into latents.
        diffusion_unet: Pretrained diffusion UNet operating in latent space.
        controlnet: ControlNet that processes image-space control inputs.
        noise_scheduler: Noise scheduler implementing ``add_noise``.
        hparams: Dictionary of hyper-parameters controlling training.
        optimizer: Optimiser factory (partial) used to construct the optimiser.
        polylr: Optional learning rate scheduler factory (partial).

    Required batch keys:
        ``latent`` or ``image`` depending on ``input_key``
        ``mask_one_hot``: Tensor (B,C,H,W,D) for segmentation channels.
        ``image_masked``: Tensor (B,C_img,H,W,D) for masked input images.
        ``modality_map``: Tensor (B,) containing modality indices.
    """

    def __init__(
        self,
        autoencoder: Optional[nn.Module],
        diffusion_unet: nn.Module,
        controlnet: nn.Module,
        noise_scheduler,
        hparams: Dict[str, Any],
        optimizer,
        polylr=None,
        demographics_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        hp_plain = _to_plain(hparams)
        self.save_hyperparameters(
            hp_plain,
            ignore=[
                "autoencoder",
                "diffusion_unet",
                "controlnet",
                "noise_scheduler",
                "optimizer",
                "polylr",
                "demographics_encoder",
            ],
        )

        # Modules
        self.autoencoder = autoencoder
        self.unet = diffusion_unet
        self.controlnet = controlnet
        self.demographics_encoder = demographics_encoder
        if noise_scheduler is None:
            raise ValueError("noise_scheduler must be provided")
        object.__setattr__(self, "noise_scheduler", noise_scheduler)

        # Factories / partials
        self.optimizer_partial = optimizer
        self.polylr_partial = polylr

        # Loss
        loss_name = str(self.hparams.get("loss", "L1")).upper()
        self.criterion = nn.MSELoss() if loss_name == "L2" else nn.L1Loss()

        # Hparams
        hp = self.hparams
        latent_source = str(hp.get("latent_source", "mu")).lower()
        if latent_source not in {"mu", "sample"}:
            raise ValueError("latent_source must be 'mu' or 'sample'")
        self.latent_source = latent_source
        # Target latent key (preferred). If omitted, fall back to legacy `input_key` behavior.
        _tk = str(hp.get("target_key", "")).strip()
        self.target_key: Optional[str] = _tk if _tk else None
        self.input_key = str(hp.get("input_key", "latent"))
        self.use_modality_mapping = bool(hp.get("use_modality_mapping", True))
        self.modality_map = {str(k): int(v) for k, v in hp.get("modality_map", {}).items()}
        modality_values = list(self.modality_map.values())
        self._modality_min = min(modality_values) if modality_values else 0
        self._modality_max = max(modality_values) if modality_values else -1
        self.num_inference_steps = int(hp.get("num_inference_steps", 30))
        self.latent_channels = int(hp.get("latent_channels", 4))
        self.conditioning_scale = float(hp.get("conditioning_scale", 1.0))

        # ControlNet conditioning builder
        self.control_keys = list(hp.get("control_keys", []) or [])
        self.control_concat_dim = int(hp.get("control_concat_dim", 1))
        self.control_apply_scale_factor = bool(hp.get("control_apply_scale_factor", True))
        self.is_mod_synthetic_key = str(hp.get("is_mod_synthetic_key", "")).strip() or None

        # Prediction type (match DiffusionModule behavior)
        _pt = hp.get("prediction_type", getattr(self.noise_scheduler, "prediction_type", "v_prediction"))
        self.prediction_type = DDPMPredictionType(str(_pt))

        # Scale factor: Must be provided
        _sf_val = hp.get("scale_factor", None)
        if _sf_val is None:
            raise ValueError("'scale_factor' must be provided in hparams")
        try:
            sf = float(_sf_val)
        except Exception as e:
            raise ValueError("'scale_factor' must be a number") from e
        if sf <= 0.0:
            raise ValueError(f"'scale_factor' must be > 0. Got {sf}.")
        self.register_buffer("_scale_factor", torch.tensor(sf, dtype=torch.float32), persistent=True)

        # Weight Decay Settings (same pattern as DiffusionModule)
        self.weight_decay_conf = hp.get("weight_decay", {}) or {}

        # EMA over ControlNet (optional)
        ema_cfg = hp.get("ema", {}) or {}
        self._ema_enabled = bool(ema_cfg.get("enabled", False))
        self._ema_use_for_sampling = bool(ema_cfg.get("use_for_sampling", True))
        self._ema_last_step: int = -1
        self._ema_loaded: bool = False
        self.ema: Optional[ExponentialMovingAverage] = None
        if self._ema_enabled:
            cfg = EMAConfig(
                decay=float(ema_cfg.get("decay", 0.9999)),
                update_every=int(ema_cfg.get("update_every", 1)),
                start_step=int(ema_cfg.get("start_step", 0)),
                store_on_cpu=bool(ema_cfg.get("store_on_cpu", False)),
                use_fp32_shadow=bool(ema_cfg.get("use_fp32_shadow", True)),
                track_all_params=bool(ema_cfg.get("track_all_params", True)),
                update_frozen=bool(ema_cfg.get("update_frozen", False)),
            )
            self.ema = ExponentialMovingAverage({"controlnet": self.controlnet}, cfg=cfg)
        
        # Freeze AE + UNet; train only ControlNet
        if self.autoencoder is not None:
            for p in self.autoencoder.parameters():
                p.requires_grad = False
            self.autoencoder.eval()
        if self.unet is not None:
            for p in self.unet.parameters():
                p.requires_grad = False
            self.unet.eval()
        if self.demographics_encoder is not None:
            for p in self.demographics_encoder.parameters():
                p.requires_grad = False
            self.demographics_encoder.eval()
        for p in self.controlnet.parameters():
            p.requires_grad = True

        # Validation aggregation buffers
        self.register_buffer("_val_loss_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_val_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("_val_loss_ema_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_val_ema_count", torch.tensor(0, dtype=torch.long), persistent=False)

        # Memory trackers
        self.train_memory_tracker: Optional[SimpleGPUMemoryTracker] = None
        self.val_memory_tracker: Optional[SimpleGPUMemoryTracker] = None

        # Total steps (populated in setup)
        self._total_steps: Optional[int] = None
        self._lr_scheduler = None

        # ------------------------------------------------------------
        # Observed sampling stats (bucket usage + synthetic selection)
        # ------------------------------------------------------------
        # Purpose: verify what the model *actually* sees after samplers + transforms
        # (bucket sampling + stream selection). This is intentionally strict: if
        # bucket labels are not present, we raise early rather than silently logging
        # zeros.
        hp = self.hparams
        self._obs_sampling_enabled: bool = bool(hp.get("log_observed_sampling", True))
        self._obs_sampling_strict: bool = bool(hp.get("observed_sampling_strict", True))
        self._obs_bucket_names: Optional[List[str]] = None
        self._obs_bucket_name_to_idx: Optional[Dict[str, int]] = None
        self._obs_modality_names: List[str] = [
            str(m).strip().lower()
            for m in (hp.get("observed_sampling_modality_order", ["t1w", "t2w", "flair", "t1ce"]) or [])
        ]
        self._obs_log_val: bool = bool(hp.get("log_observed_sampling_val", True))
        self._obs_bucket_counts_train: Optional[torch.Tensor] = None
        self._obs_syn_sum_train: Optional[torch.Tensor] = None
        self._obs_bucket_counts_val: Optional[torch.Tensor] = None
        self._obs_syn_sum_val: Optional[torch.Tensor] = None


    # EMA helpers (mirrors DiffusionModule)
    @contextlib.contextmanager
    def ema_scope(self, enabled: bool = True):
        """Temporarily swap ControlNet params to EMA weights within the scope."""
        if not (enabled and self._ema_enabled and self.ema is not None):
            yield
            return
        self.ema.store()  # store all tracked modules
        self.ema.copy_to()  # copy EMA weights to all tracked modules
        try:
            yield
        finally:
            self.ema.restore()  # restore original weights for all tracked modules


    # Optimizer / weight decay utilities (mirrors DiffusionModule)
    def _get_weight_decay_from_partial(self) -> float:
        kw = getattr(self.optimizer_partial, "keywords", {}) or {}
        wd = kw.get("weight_decay", 0.0)
        try:
            return float(wd)
        except Exception:
            return 0.0

    def _assert_weight_decay_config(self):
        wd_enabled = bool(self.weight_decay_conf.get("enabled", False))
        split_groups = bool(self.weight_decay_conf.get("split_param_groups", True))

        opt_wd = self._get_weight_decay_from_partial()
        opt_tar = self._get_opt_target_name()

        if wd_enabled and split_groups:
            wrong_optim = (opt_tar != "torch.optim.adamw.AdamW" and not opt_tar.endswith(".AdamW"))
            if wrong_optim or opt_wd <= 0.0:
                raise ValueError("Optimizer is configured incorrectly!")

    def _get_opt_target_name(self) -> str:
        """Best-effort name of the optimizer class/function used by optimizer_partial."""
        opt = getattr(self, "optimizer_partial", None)
        if isinstance(opt, functools.partial):
            func = opt.func
            return f"{getattr(func, '__module__', '')}.{getattr(func, '__name__', func.__class__.__name__)}"
        return str(type(opt))

    def _build_param_groups(self, modules: List[Tuple[str, nn.Module]], weight_decay: float):
        """
        Create two param groups:
        - decay: params that get weight decay
        - no_decay: biases + norm params + embeddings
        """
        norm_types = (
            nn.LayerNorm,
            nn.GroupNorm,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        )
        no_decay_ids = set()
        all_named_params = []

        for prefix, m in modules:
            if m is None:
                continue

            # collect named params (for final grouping)
            for n, p in m.named_parameters():
                if p is None or not p.requires_grad:
                    continue
                full_name = f"{prefix}.{n}" if prefix else n
                all_named_params.append((full_name, p))

                # bias -> no decay
                if n.endswith("bias"):
                    no_decay_ids.add(id(p))

            # norm modules -> no decay
            for _, sub in m.named_modules():
                if isinstance(sub, norm_types):
                    for _, p in sub.named_parameters(recurse=False):
                        if p is not None and p.requires_grad:
                            no_decay_ids.add(id(p))

            # embeddings often excluded too
            for _, sub in m.named_modules():
                if isinstance(sub, nn.Embedding):
                    for _, p in sub.named_parameters(recurse=False):
                        if p is not None and p.requires_grad:
                            no_decay_ids.add(id(p))

        decay_params, no_decay_params = [], []
        seen = set()
        for _, p in all_named_params:
            if id(p) in seen:
                continue
            seen.add(id(p))
            (no_decay_params if id(p) in no_decay_ids else decay_params).append(p)

        groups = []
        if decay_params:
            groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return groups

    def _log_learning_rates(self, when: str, opt: Optional[torch.optim.Optimizer] = None) -> None:
        optimizers = []
        if opt is not None:
            optimizers = [opt]
        else:
            existing_opts = self.optimizers()
            if existing_opts is None:
                raise ValueError("Unable to find Optimizers")
            optimizers = existing_opts if isinstance(existing_opts, Sequence) else [existing_opts]

        lr_msgs = []
        for opt_idx, optimizer in enumerate(optimizers):
            if optimizer is None:
                continue
            group_lrs = []
            for pg in getattr(optimizer, "param_groups", []):
                lr = pg.get("lr")
                try:
                    group_lrs.append(f"{float(lr):.6g}")
                except (TypeError, ValueError):
                    group_lrs.append(str(lr))
            if group_lrs:
                lr_msgs.append(f"opt{opt_idx}: {', '.join(group_lrs)}")

        if lr_msgs:
            _LOG.info(f"[ControlNetModule] Learning rate ({when}): {' | '.join(lr_msgs)}")


    def _encode_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latents using the autoencoder if provided.  Otherwise
        return the input as latents directly.
        """
        if self.autoencoder is None:
            return images
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=False):
                enc = getattr(self.autoencoder, "encode", None)
                if callable(enc):
                    z_mu, z_sigma = enc(images.float())
                else:
                    _, z_mu, z_sigma = self.autoencoder(images.float())
        if self.latent_source == "sample" and z_sigma is not None:
            eps = torch.randn_like(z_sigma)
            z = z_mu + z_sigma * eps
        else:
            z = z_mu
        return z

    def _get_scale_factor(self, z: torch.Tensor) -> torch.Tensor:
        # Stored as a persistent buffer for correct checkpointing / device moves.
        return self._scale_factor.to(device=z.device, dtype=z.dtype)

    def _compute_z(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Extract the latent tensor to denoise.

        Preferred path (generic):
          - if ``hparams.target_key`` is set, we read the latent directly from
            ``batch[target_key]``.

        Legacy path (backwards compatible):
          - if ``input_key == 'latent'`` use ``batch['latent']``.
          - else encode ``batch['image']`` with the autoencoder.
        """
        if self.target_key:
            if self.target_key not in batch:
                raise KeyError(f"Batch is missing target_key={self.target_key!r}")
            z = batch[self.target_key]
            if not isinstance(z, torch.Tensor):
                raise TypeError(f"batch[{self.target_key!r}] must be a torch.Tensor")
            z = z.to(self.device, non_blocking=True)
            return z * self._get_scale_factor(z)
        
        
        raise ValueError("Target Key Is Not Set, TODO!")
        #TODO: This following was the fallback logic, Maybe not required. Keeping for now.

        if self.input_key == "latent":  # TODO: MAYBE CONVERT INTO ENUM
            z = batch["latent"]
            if not isinstance(z, torch.Tensor):
                z = torch.tensor(z)
            z = z.to(self.device, non_blocking=True)
        else:
            if "image" not in batch:
                raise ValueError("Autoencoder is used when input_key!='latent', but 'image' is missing from batch")
            img = batch["image"]
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
            img = img.to(self.device, non_blocking=True)
            z = self._encode_latent(img)

        return z * self._get_scale_factor(z)

    def _apply_noise(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Sample timesteps and add noise to latents using the scheduler. """
        noise = torch.randn_like(z)
        if isinstance(self.noise_scheduler, RFlowScheduler):
            timesteps = self.noise_scheduler.sample_timesteps(z)
        else:
            num_tr = int(getattr(self.noise_scheduler, "num_train_timesteps"))
            timesteps = torch.randint(0, num_tr, (z.shape[0],), device=z.device).long()

        z_noisy = self.noise_scheduler.add_noise(original_samples=z, noise=noise, timesteps=timesteps)
        return timesteps, noise, z_noisy

    def _target_for_prediction_type(
        self,
        pred_type: DDPMPredictionType,
        z: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the denoising target consistent with DiffusionModule."""
        if pred_type == DDPMPredictionType.EPSILON:
            return noise
        if pred_type == DDPMPredictionType.SAMPLE:
            return z
        if pred_type == DDPMPredictionType.V_PREDICTION:
            if isinstance(self.noise_scheduler, DDPMScheduler):
                return self.noise_scheduler.get_velocity(sample=z, noise=noise, timesteps=timesteps)
            if isinstance(self.noise_scheduler, RFlowScheduler):
                return z - noise
            raise TypeError(f"V_PREDICTION unsupported for {type(self.noise_scheduler).__name__}")
        raise ValueError(f"Unknown prediction type: {pred_type}")

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        return self.criterion(pred.float(), target.float())

    def _prep_condition_tensors(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if getattr(self.unet, "num_class_embeds", 0):
            if "modality_map" not in batch or not torch.is_tensor(batch["modality_map"]):
                raise ValueError("Missing 'modality_map' in batch; ensure conditioning transform is enabled")
            return {"class_labels": batch["modality_map"].to(self.device, non_blocking=True)}
        else:
            # Passing class labels enforced for now (will be removed)
            raise KeyError("Missing 'class_labels' for class-conditioned sampling")

    def _encode_demographics(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Encode demographics for UNet conditioning, mirroring DiffusionModule.

        This is *not* passed into ControlNet: the ControlNet branch is kept generic
        and independent from metadata encoders.
        """
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
            raise ValueError(f"demo_values and demo_missing shape mismatch: {demo_values.shape} vs {demo_missing.shape}")

        demo_values = demo_values.to(self.device, non_blocking=True)
        demo_missing = demo_missing.to(self.device, non_blocking=True)

        # Black-box encoder: (B, D) + (B, D) -> (B, D_dem)
        with torch.no_grad():
            dem_emb = self.demographics_encoder(demo_values, demo_missing)

        if not torch.is_tensor(dem_emb):
            raise TypeError("demographics_encoder must return a torch.Tensor")
        if dem_emb.ndim != 2:
            raise ValueError(f"demographics_embedding must be 2D (B,D), got {dem_emb.shape}")
        if dem_emb.shape[0] != demo_values.shape[0]:
            raise ValueError("Batch size mismatch in demographics embedding")
        return dem_emb

    def _get_is_mod_synthetic(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Return per-sample synthetic-modality flags (B, K) if present."""
        if self.is_mod_synthetic_key and self.is_mod_synthetic_key in batch:
            v = batch[self.is_mod_synthetic_key]
            if torch.is_tensor(v):
                return v.to(self.device, non_blocking=True).float()
        raise KeyError("Missing key 'is_mod_synthetic' from batch")

    def _prep_controlnet_cond(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Build the conditioning tensor passed to ControlNet.

        Generic path (preferred):
          - if ``hparams.control_keys`` is a non-empty list, concatenate those tensors.
            This is what you want for modality translation in latent space.

        Notes:
          - set ``hparams.control_apply_scale_factor=true`` when your control inputs are *latents*.
          - keep ``control_apply_scale_factor=false`` when your control inputs are in image-space.
        """
        if self.control_keys:
            xs: list[torch.Tensor] = []
            for k in self.control_keys:
                if k not in batch:
                    raise KeyError(f"Missing control key {k!r} in batch")
                t = batch[k]
                if not torch.is_tensor(t):
                    raise TypeError(f"Control key {k!r} must be a torch.Tensor")
                t = t.to(self.device, non_blocking=True)
                if self.control_apply_scale_factor:
                    t = t * self._get_scale_factor(t)
                xs.append(t)
            return torch.cat(xs, dim=self.control_concat_dim)

        raise KeyError("Missing ControlNet conditioning.")


    def setup(self, stage: Optional[str] = None) -> None:
        if hasattr(super(), "setup"):
            super().setup(stage)  # Lightning setup()

        # Warn if VAE is not loaded
        if self.autoencoder is None:
            warnings.warn("No autoencoder provided - latents will pass through during decode")
        else:
            has_ae_spec = any(s.get("target") == "autoencoder" for s in self.hparams.get("weight_loads", []))
            if has_ae_spec and "autoencoder" not in self.get_loaded_models():
                raise ValueError("Autoencoder specified in weight_loads but not loaded by StateDictLoaderMixin")

        # Warn if Diffusion UNET is not loaded
        if self.unet is None:
            warnings.warn("No diffusion UNET provided - please provide trained UNET weights")
        else:
            has_unet_spec = any(s.get("target") == "unet" for s in self.hparams.get("weight_loads", []))
            if has_unet_spec and "unet" not in self.get_loaded_models():
                raise ValueError("UNet specified in weight_loads but not loaded by StateDictLoaderMixin")

        # Compute total iterations for PolynomialLR: epochs * steps/epoch
        if stage in (None, "fit"):
            est = getattr(self.trainer, "estimated_stepping_batches", None)
            if est is None:
                train_batches = len(self.trainer.datamodule.train_dataloader())
                est = self.trainer.max_epochs * train_batches
            self._total_steps = int(est)

        # If EMA is enabled and we did not restore EMA from a checkpoint, reset
        # EMA to match the (possibly externally loaded) ControlNet weights.
        if self._ema_enabled and self.ema is not None and (not self._ema_loaded) and stage in (None, "fit"):
            self.ema.reset()

        # Ensure EMA shadows reside on the correct device.
        if self._ema_enabled and self.ema is not None:
            self.ema.to(self.device)

    def on_fit_start(self) -> None:
        if self.train_memory_tracker is None:
            self.train_memory_tracker = SimpleGPUMemoryTracker(device=self.device)
        if self.val_memory_tracker is None:
            self.val_memory_tracker = SimpleGPUMemoryTracker(device=self.device)
        if self._obs_sampling_enabled:
            self._init_observed_sampling_state()


    def configure_optimizers(self):
        wd_enabled = bool(self.weight_decay_conf.get("enabled", False))
        split_groups = bool(self.weight_decay_conf.get("split_param_groups", True))

        if wd_enabled and split_groups:
            self._assert_weight_decay_config()
            weight_decay = self._get_weight_decay_from_partial()
            param_groups = self._build_param_groups([("controlnet", self.controlnet)], weight_decay=weight_decay)
            opt = self.optimizer_partial(params=param_groups)
        else:
            opt = self.optimizer_partial(params=self.controlnet.parameters())

        if self.polylr_partial is not None:
            if self._total_steps is None:
                raise RuntimeError("total steps not computed; check setup('fit').")
            sched = self.polylr_partial(optimizer=opt, total_iters=self._total_steps)
            self._lr_scheduler = sched
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
            }

        # Log learning rates at start
        self._log_learning_rates("configure", opt)
        return opt

    def on_train_epoch_start(self) -> None:
        if self.train_memory_tracker is not None:
            self.train_memory_tracker.reset_peak()

        # Enable Training Mode 
        self.controlnet.train()

        if self._obs_sampling_enabled:
            self._reset_observed_sampling_epoch(stage="train")

    def on_validation_epoch_start(self) -> None:
        if self.val_memory_tracker is not None:
            self.val_memory_tracker.reset_peak()


        if self._obs_sampling_enabled and getattr(self, "_obs_log_val", True):
            self._reset_observed_sampling_epoch(stage="val")



    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        z = self._compute_z(batch)  # Compute latents
        bs = z.size(0)
        timesteps, noise, z_noisy = self._apply_noise(z)

        # ControlNet conditioning input
        ctrl = self._prep_controlnet_cond(batch)

        # Class labels (optional)
        cl_dict = self._prep_condition_tensors(batch)
        

        expected_c = getattr(self.controlnet, "conditioning_embedding_in_channels", None)
        if expected_c is None:
            raise AttributeError("ControlNet expected in-channels not found (expose conditioning_embedding_in_channels).")

        if ctrl is None:
            raise ValueError("ControlNet conditioning tensor `ctrl` is None.")

        if int(ctrl.shape[1]) != int(expected_c):
            raise ValueError(f"ctrl C={int(ctrl.shape[1])} != ControlNet expected C={int(expected_c)}")


        # Demographics embedding for UNet (optional)
        dem_emb = self._encode_demographics(batch)

        # Optional synthetic-modality flags for ControlNet
        is_syn = self._get_is_mod_synthetic(batch)

        # ControlNet forward (robust to signature diffs)
        down_samples, mid_sample = self.controlnet(
            x=z_noisy,
            timesteps=timesteps,
            controlnet_cond=ctrl,
            conditioning_scale=self.conditioning_scale,
            context=None,
            class_labels=cl_dict.get('class_labels'),
            is_mod_synthetic=is_syn,
        )

        # UNet forward
        unet_inputs: Dict[str, Any] = {
            "x": z_noisy,
            "timesteps": timesteps,
            "down_block_additional_residuals": down_samples,
            "mid_block_additional_residual": mid_sample,
        }
        unet_inputs.update(cl_dict)
        if dem_emb is not None:
            unet_inputs["demographics_embedding"] = dem_emb
        # Forward through diffusion UNet
        pred = self.unet(**unet_inputs)

        target = self._target_for_prediction_type(
            pred_type=self.prediction_type,
            z=z,
            noise=noise,
            timesteps=timesteps,
        )
        loss = self.compute_loss(pred, target, batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs, logger=False)
        self.log("train/loss_epoch", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs, logger=True)

        # TB scalar if available
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            log_every_n_steps = int(getattr(self.trainer, "log_every_n_steps", 100))
            tb_writer = self._tb_writer()
            step_ix = int(self.global_step) + 1
            if (
                log_every_n_steps > 0
                and (step_ix % log_every_n_steps == 0)
                and self.trainer.is_global_zero
                and tb_writer is not None
            ):
                tb_writer.add_scalar("train/loss", float(loss.detach()), step_ix)        
        
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        z = self._compute_z(batch)
        bs = z.size(0)
        timesteps, noise, z_noisy = self._apply_noise(z)

        ctrl = self._prep_controlnet_cond(batch)
        cl_dict = self._prep_condition_tensors(batch)
        dem_emb = self._encode_demographics(batch)
        is_syn = self._get_is_mod_synthetic(batch)

        down_samples, mid_sample = self.controlnet(
            x=z_noisy,
            timesteps=timesteps,
            controlnet_cond=ctrl,
            conditioning_scale=self.conditioning_scale,
            context=None,
            class_labels=cl_dict.get('class_labels'),
            is_mod_synthetic=is_syn,
        )

        unet_inputs = {
            "x": z_noisy,
            "timesteps": timesteps,
            "down_block_additional_residuals": down_samples,
            "mid_block_additional_residual": mid_sample,
        }
        unet_inputs.update(cl_dict)
        if dem_emb is not None:
            unet_inputs["demographics_embedding"] = dem_emb
        pred = self.unet(**unet_inputs)

        target = self._target_for_prediction_type(
            pred_type=self.prediction_type,
            z=z,
            noise=noise,
            timesteps=timesteps,
        )
        loss = self.compute_loss(pred, target, batch)
        # accumulate for true epoch-weighted average
        self._val_loss_sum += loss.detach() * bs
        self._val_count += bs

        # Optional EMA validation pass (EMA over ControlNet parameters).
        if self._ema_enabled and (self.ema is not None):
            with self.ema_scope(enabled=True):
                down_s_ema, mid_s_ema = self.controlnet(
                    x=z_noisy,
                    timesteps=timesteps,
                    controlnet_cond=ctrl,
                    conditioning_scale=self.conditioning_scale,
                    context=None,
                    class_labels=cl_dict.get('class_labels'),
                    is_mod_synthetic=is_syn,
                )
                unet_inputs_ema = dict(unet_inputs)
                unet_inputs_ema["down_block_additional_residuals"] = down_s_ema
                unet_inputs_ema["mid_block_additional_residual"] = mid_s_ema
                pred_ema = self.unet(**unet_inputs_ema)
            loss_ema = self.compute_loss(pred_ema, target, batch)
            self._val_loss_ema_sum += loss_ema.detach() * bs
            self._val_ema_count += bs
        return loss

    def on_validation_epoch_end(self):
        # Global (across ranks) weighted mean: sum(loss * n) / sum(n)
        sum_all = self.trainer.strategy.reduce(self._val_loss_sum, reduce_op="sum")
        cnt_all = self.trainer.strategy.reduce(self._val_count, reduce_op="sum")
        val_loss = sum_all / cnt_all.clamp_min(1).float()

        # log once per epoch;
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=False)
        self.log("val_total", val_loss, prog_bar=False, logger=False, sync_dist=False)  # alias for filename/monitor

        if self._ema_enabled and (self.ema is not None):
            sum_ema = self.trainer.strategy.reduce(self._val_loss_ema_sum, reduce_op="sum")
            cnt_ema = self.trainer.strategy.reduce(self._val_ema_count, reduce_op="sum")
            val_loss_ema = sum_ema / cnt_ema.clamp_min(1).float()
            self.log("val/loss_ema", val_loss_ema, prog_bar=True, sync_dist=False)

        # reset for next epoch
        self._val_loss_sum.zero_()
        self._val_count.zero_()
        self._val_loss_ema_sum.zero_()
        self._val_ema_count.zero_()

        if self._obs_sampling_enabled and getattr(self, "_obs_log_val", True):
            self._log_observed_sampling(stage="val")

        if self.val_memory_tracker is not None:
            _LOG.info(self.val_memory_tracker.record_peak_memory(self.current_epoch, "VAL"))

    def on_train_epoch_end(self) -> None:
        if self.train_memory_tracker is not None:
            _LOG.info(self.train_memory_tracker.record_peak_memory(self.current_epoch, "TRAIN"))

        if self._obs_sampling_enabled:
            self._log_observed_sampling(stage="train")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # ------------------------------------------------------------
        # Observed sampling accumulation (STRICT)
        # ------------------------------------------------------------
        if self._obs_sampling_enabled:
            self._accumulate_observed_sampling(batch, stage="train")

        # Update EMA after optimizer step (global_step only increments when an optimizer step occurs).
        if not (self._ema_enabled and self.ema is not None):
            return
        step = int(self.global_step)
        if step <= 0:
            return
        if step == self._ema_last_step:
            return
        self.ema.update(step=step)
        self._ema_last_step = step

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Observed sampling accumulation for VAL (STRICT)
        if self._obs_sampling_enabled and getattr(self, "_obs_log_val", True):
            self._accumulate_observed_sampling(batch, stage="val")

    def on_save_checkpoint(self, checkpoint):
        # Persist scale_factor for safety/consistency
        checkpoint["scale_factor"] = self._scale_factor.detach().float().cpu()

        if self._ema_enabled and self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.ema_state_dict()
            checkpoint["ema_last_step"] = int(self._ema_last_step)

    def on_load_checkpoint(self, checkpoint):
        sf = checkpoint.get("scale_factor", None)
        if sf is not None:
            try:
                sf_t = sf if torch.is_tensor(sf) else torch.tensor(float(sf))
                self._scale_factor.copy_(sf_t.to(self._scale_factor.device, dtype=self._scale_factor.dtype))
                _LOG.info(f"[ControlNetModule] Restored scale_factor={float(self._scale_factor.item()):.6f}")
            except Exception as e:
                _LOG.warning("[ControlNetModule] Failed to restore scale_factor: %s", e)

        if not (self._ema_enabled and self.ema is not None):
            return

        ema_state = checkpoint.get("ema", None)
        if ema_state is None:
            ema_sd = checkpoint.get("ema_state_dict", None)
            if ema_sd is not None:
                ema_state = {"ema_state_dict": ema_sd}

        if ema_state is not None:
            try:
                self.ema.load_state_dict(ema_state)
                self._ema_last_step = int(checkpoint.get("ema_last_step", -1))
                self._ema_loaded = True
                _LOG.info("[ControlNetModule] Restored EMA state.")
            except Exception as e:
                _LOG.warning("[ControlNetModule] Failed to restore EMA state: %s", e)

    # Observed sampling: init / accumulate / log
    def _init_observed_sampling_state(self) -> None:
        """Initialize per-epoch counters and bucket name mapping.

        This is strict by default: if bucket names cannot be discovered from the
        datamodule, we raise with a clear message.
        """
        if not self._obs_sampling_enabled:
            return
        if self._obs_bucket_counts_train is not None:
            return  # already initialized

        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            raise RuntimeError(
                "[ObservedSampling] trainer.datamodule is None at on_fit_start. "
                "This usually indicates the hook is being called before the datamodule is attached."
            )

        b2i = getattr(dm, "_bucket_to_indices", None)
        if not isinstance(b2i, dict):
            raise RuntimeError(
                "[ObservedSampling] datamodule does not expose _bucket_to_indices (dict). "
                "Expected BrainScapePairedDataModule-style bucketing. "
                f"Got: {type(b2i)}"
            )
        train_buckets = b2i.get("train", None)
        if not isinstance(train_buckets, dict) or not train_buckets:
            raise RuntimeError(
                "[ObservedSampling] datamodule._bucket_to_indices['train'] is missing or empty. "
                "Check that your dataset config defines buckets.rules and that setup('fit') ran."
            )

        bucket_names = list(train_buckets.keys())
        if not bucket_names:
            raise RuntimeError(
                "[ObservedSampling] No bucket names discovered from datamodule._bucket_to_indices['train']."
            )

        # Canonicalize mapping keys to avoid whitespace/casing surprises.
        bucket_names_sorted = sorted([str(b) for b in bucket_names])
        self._obs_bucket_names = bucket_names_sorted
        self._obs_bucket_name_to_idx = {
            str(b).strip().lower(): i for i, b in enumerate(bucket_names_sorted)
        }

        n_b = len(bucket_names_sorted)
        n_m = len(self._obs_modality_names)
        if n_m <= 0:
            raise RuntimeError("[ObservedSampling] observed_sampling_modality_order is empty.")

        # Keep counters on the module device so updates are cheap.
        dev = self.device
        self._obs_bucket_counts_train = torch.zeros(n_b, device=dev, dtype=torch.float32)
        self._obs_syn_sum_train = torch.zeros(n_b, n_m, device=dev, dtype=torch.float32)
        self._obs_bucket_counts_val = torch.zeros(n_b, device=dev, dtype=torch.float32)
        self._obs_syn_sum_val = torch.zeros(n_b, n_m, device=dev, dtype=torch.float32)

        if self.trainer.is_global_zero:
            msg = (
                "[ObservedSampling] enabled=True strict="
                f"{self._obs_sampling_strict} buckets={self._obs_bucket_names} "
                f"modalities={self._obs_modality_names} key_is_syn={self.is_mod_synthetic_key}"
            )
            print(msg)
            _LOG.info(msg)

    def _reset_observed_sampling_epoch(self, *, stage: str = "train") -> None:
        if not self._obs_sampling_enabled:
            return
        st = str(stage).lower().strip()
        if st == "train":
            counts = self._obs_bucket_counts_train
            syn = self._obs_syn_sum_train
        elif st in ("val", "valid", "validation"):
            counts = self._obs_bucket_counts_val
            syn = self._obs_syn_sum_val
        else:
            raise ValueError(f"[ObservedSampling] Unknown stage={stage!r}")
        if counts is None or syn is None:
            raise RuntimeError("[ObservedSampling] state not initialized. Expected _init_observed_sampling_state() to run.")
        counts.zero_()
        syn.zero_()

    @torch.no_grad()
    def _accumulate_observed_sampling(self, batch: Dict[str, Any], *, stage: str = "train") -> None:
        """Accumulate empirical bucket frequencies and synthetic selection rates."""
        if not self._obs_sampling_enabled:
            return
        if self._obs_bucket_name_to_idx is None:
            raise RuntimeError("[ObservedSampling] state not initialized. Expected _init_observed_sampling_state() to run.")
        st = str(stage).lower().strip()
        if st == "train":
            bucket_counts = self._obs_bucket_counts_train
            syn_sum = self._obs_syn_sum_train
        elif st in ("val", "valid", "validation"):
            bucket_counts = self._obs_bucket_counts_val
            syn_sum = self._obs_syn_sum_val
        else:
            raise ValueError(f"[ObservedSampling] Unknown stage={stage!r}")
        if bucket_counts is None or syn_sum is None:
            raise RuntimeError("[ObservedSampling] state not initialized. Expected _init_observed_sampling_state() to run.")

        if not isinstance(batch, dict):
            raise TypeError(f"[ObservedSampling] batch must be a dict; got {type(batch)}")

        buckets = batch.get("bucket", None)
        if buckets is None:
            raise KeyError(
                "[ObservedSampling] batch is missing key 'bucket'. "
                "Fix: ensure meta_keys includes 'bucket' and ChooseStreamForModalitiesd.drop_bucket=false. "
                f"batch_keys={list(batch.keys())}"
            )

        # Infer batch size from the first tensor in the batch.
        bs = None
        for v in batch.values():
            if torch.is_tensor(v):
                bs = int(v.shape[0])
                break
        if bs is None:
            raise RuntimeError("[ObservedSampling] Cannot infer batch size (no tensor values found in batch).")

        if not isinstance(buckets, (list, tuple)):
            # Strict by design — MONAI collate should produce list/tuple for string metadata.
            raise TypeError(
                "[ObservedSampling] Expected batch['bucket'] to be list/tuple of strings (len=B). "
                f"Got type={type(buckets)} repr={repr(buckets)[:200]}"
            )

        bucket_list = [str(b).strip().lower() for b in buckets]
        if len(bucket_list) != bs:
            raise ValueError(
                "[ObservedSampling] bucket list length != batch size. "
                f"len(bucket)={len(bucket_list)} bs={bs}"
            )

        idx_py = []
        unknown = []
        for b in bucket_list:
            j = self._obs_bucket_name_to_idx.get(b, None)
            if j is None:
                unknown.append(b)
                j = -1
            idx_py.append(j)

        if unknown:
            raise KeyError(
                "[ObservedSampling] Encountered unknown bucket name(s) in batch. "
                f"unknown={sorted(set(unknown))} known={list(self._obs_bucket_name_to_idx.keys())}"
            )

        idx = torch.tensor(idx_py, device=bucket_counts.device, dtype=torch.long)
        ones = torch.ones(bs, device=bucket_counts.device, dtype=torch.float32)
        bucket_counts.index_add_(0, idx, ones)

        # Synthetic flags accumulation (per modality)
        if self.is_mod_synthetic_key is None:
            return
        if self.is_mod_synthetic_key not in batch:
            raise KeyError(
                "[ObservedSampling] is_mod_synthetic_key is set but missing in batch. "
                f"key={self.is_mod_synthetic_key} batch_keys={list(batch.keys())}"
            )

        is_syn = batch.get(self.is_mod_synthetic_key, None)
        if is_syn is None:
            raise RuntimeError("[ObservedSampling] batch[is_mod_synthetic_key] is None")
        if not torch.is_tensor(is_syn):
            raise TypeError(
                "[ObservedSampling] Expected is_mod_synthetic tensor in batch. "
                f"Got type={type(is_syn)}"
            )
        if is_syn.dim() != 2:
            raise ValueError(
                "[ObservedSampling] Expected is_mod_synthetic to have shape (B, M). "
                f"Got shape={tuple(is_syn.shape)}"
            )
        if int(is_syn.shape[0]) != bs:
            raise ValueError(
                "[ObservedSampling] is_mod_synthetic batch dim mismatch. "
                f"shape[0]={int(is_syn.shape[0])} bs={bs}"
            )

        n_mod = len(self._obs_modality_names)
        if int(is_syn.shape[1]) != n_mod:
            raise ValueError(
                "[ObservedSampling] is_mod_synthetic modality dim mismatch. "
                f"shape[1]={int(is_syn.shape[1])} expected={n_mod} "
                f"order={self._obs_modality_names}"
            )

        syn_sum.index_add_(0, idx, is_syn.float())

    def _log_observed_sampling(self, *, stage: str = "train") -> None:
        """Reduce across ranks and log/print observed bucket + synthetic rates."""
        if not self._obs_sampling_enabled:
            return
        if self._obs_bucket_names is None:
            raise RuntimeError("[ObservedSampling] state not initialized. Expected _init_observed_sampling_state() to run.")
        st = str(stage).lower().strip()
        if st == "train":
            bucket_counts = self._obs_bucket_counts_train
            syn_sum = self._obs_syn_sum_train
        elif st in ("val", "valid", "validation"):
            bucket_counts = self._obs_bucket_counts_val
            syn_sum = self._obs_syn_sum_val
        else:
            raise ValueError(f"[ObservedSampling] Unknown stage={stage!r}")
        if bucket_counts is None or syn_sum is None:
            raise RuntimeError("[ObservedSampling] state not initialized. Expected _init_observed_sampling_state() to run.")

        # DDP-safe reduction
        counts = self.trainer.strategy.reduce(bucket_counts, reduce_op="sum")
        syn_sum = self.trainer.strategy.reduce(syn_sum, reduce_op="sum")

        total = float(counts.sum().item())
        if total <= 0.0:
            msg = (
                f"[ObservedSampling] epoch={int(self.current_epoch)} total=0 (no bucket labels accumulated). "
                "This means the accumulation hook never saw 'bucket' in the train batches. "
                "Most common causes: (1) your LightningModule hook signature was wrong so it never ran, "
                "or (2) 'bucket' got dropped before collation (meta_keys missing or drop_bucket=true)."
            )
            # if self._obs_sampling_strict:
            #     raise RuntimeError(msg)
            print(msg)
            _LOG.warning(msg)
            return

        probs = counts / counts.sum()
        denom = counts.clamp_min(1.0).unsqueeze(1)
        syn_rate = syn_sum / denom

        if self.trainer.is_global_zero:
            header = "bucket	count	prob	" + "	".join([f"p_syn({m})" for m in self._obs_modality_names])
            print("[ObservedSampling:%s] epoch=%d" % (st, int(self.current_epoch)))
            print(header)
            for i, b in enumerate(self._obs_bucket_names):
                row = [
                    str(b),
                    str(int(counts[i].item())),
                    f"{float(probs[i].item()):.4f}",
                ] + [f"{float(syn_rate[i, j].item()):.4f}" for j in range(len(self._obs_modality_names))]
                print("	".join(row))

        # Logger scalars (kept small)
        # for i, b in enumerate(self._obs_bucket_names):
        #     b_tag = str(b).replace("/", "_")
        #     self.log(f"obs_{st}/bucket_prob/{b_tag}", probs[i], prog_bar=False, logger=True, sync_dist=False)
        #     for j, m in enumerate(self._obs_modality_names):
        #         self.log(f"obs_{st}/p_syn/{b_tag}/{m}", syn_rate[i, j], prog_bar=False, logger=True, sync_dist=False)

    def _tb_writer(self):
        logs = self.trainer.loggers if getattr(self.trainer, "loggers", None) else (
            [self.trainer.logger] if getattr(self.trainer, "logger", None) else []
        )
        for lg in logs:
            if isinstance(lg, pl.loggers.TensorBoardLogger):
                return lg.experiment
        return None

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor via the autoencoder if available."""
        if self.autoencoder is None:
            return z
        if "autoencoder" not in self.get_loaded_models():
            raise ValueError("Autoencoder is not loaded - Cannot decode the latent image")
        dec = getattr(self.autoencoder, "decode", None)
        if callable(dec):
            return dec(z)
        raise ValueError("Autoencoder is missing the callable function decode")



    def _validate_labels(self, labels: torch.Tensor) -> None:
        """Validate that modality indices are within the configured range."""
        if labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("class_labels tensor must contain integers")
        if (labels < self._modality_min).any() or (labels > self._modality_max).any():
            raise ValueError(
                f"Invalid class label(s) {labels.tolist()}; valid range: [{self._modality_min}, {self._modality_max}]"
            )

    @torch.no_grad()
    def sample_latent_controlled(
        self,
        output_size: Tuple[int, int, int],  # (Z, Y, X)
        ctrl: torch.Tensor,                 # (B, C_ctrl, Z, Y, X)  e.g., cat(image_masked, mask_one_hot) to match training
        class_labels: Optional[torch.Tensor] = None,
        demographics_embedding: Optional[torch.Tensor] = None,
        is_mod_synthetic: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        DDPM/RFlow sampling loop with ControlNet residual injection.
        Returns latent tensor in *scaled* latent space: (B, C_latent, Z, Y, X).
        """
        device = self.device
        b = ctrl.shape[0]
        z, y, x = output_size
        img = torch.randn((b, self.latent_channels, z, y, x), device=device)

        # Enable Evaluation Mode
        self.controlnet.eval()
        
        # Set timesteps
        steps = int(num_inference_steps or self.num_inference_steps)
        if isinstance(self.noise_scheduler, RFlowScheduler):
            self.noise_scheduler.set_timesteps(
                device=device,
                num_inference_steps=steps,
                input_img_size_numel=torch.tensor(output_size, device=device).prod().item(),
            )
        else:
            self.noise_scheduler.set_timesteps(num_inference_steps=steps)
            if isinstance(self.noise_scheduler, DDPMScheduler) and steps < self.noise_scheduler.num_train_timesteps:
                warnings.warn(
                    "* WARNING: Using DDPMScheduler with num_inference_steps "
                    f"{steps} < num_train_timesteps={self.noise_scheduler.num_train_timesteps}"
                )

        # Checks
        if getattr(self.unet, "num_class_embeds", 0):
            if class_labels is None:
                raise ValueError("Missing 'class_labels' for class-conditioned sampling")
            if class_labels.numel() != b:
                raise ValueError(f"#class_labels ({class_labels.numel()}) must match batch size ({b})")
            class_labels = class_labels.to(device)

        # Sampling
        all_t = self.noise_scheduler.timesteps.to(device)
        with self.ema_scope(enabled=self._ema_use_for_sampling):
            for i, t in enumerate(all_t):
                t_val = t.item()
                next_t_val = all_t[i + 1].item() if i + 1 < len(all_t) else 0
                t_batch = torch.full((b,), t_val, device=device, dtype=all_t.dtype)

                # ControlNet forward
                cn_kwargs: Dict[str, Any] = {
                    "x": img,
                    "timesteps": t_batch,
                    "controlnet_cond": ctrl,
                    "conditioning_scale": self.conditioning_scale,
                    "context": None,
                }
                if getattr(self.unet, "num_class_embeds", 0):
                    cn_kwargs["class_labels"] = class_labels
                if is_mod_synthetic is not None:
                    cn_kwargs["is_mod_synthetic"] = is_mod_synthetic

                down_res, mid_res = self.controlnet(**cn_kwargs)

                # UNet forward with residuals
                unet_kwargs: Dict[str, Any] = {
                    "x": img,
                    "timesteps": t_batch,
                    "down_block_additional_residuals": down_res,
                    "mid_block_additional_residual": mid_res,
                }
                if getattr(self.unet, "num_class_embeds", 0):
                    unet_kwargs["class_labels"] = class_labels
                if demographics_embedding is not None:
                    unet_kwargs["demographics_embedding"] = demographics_embedding

                pred = self.unet(**unet_kwargs)

                # Scheduler step
                if isinstance(self.noise_scheduler, RFlowScheduler):
                    img, _ = self.noise_scheduler.step(pred, t_val, img, next_t_val)
                else:
                    img, _ = self.noise_scheduler.step(pred, t_val, img)

        return img  # scaled latent

    @torch.no_grad()
    def _run_inference(
        self,
        x: torch.Tensor,                      # only used for spatial size
        class_labels: Optional[torch.Tensor], # mapped and on device (or None)
        ctrl: torch.Tensor,                   # concatenated control: matches training channel order
        demographics_embedding: Optional[torch.Tensor] = None,
        is_mod_synthetic: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a *new* sample using ControlNet-guided sampling.
        """
        if x.dim() != 5:
            raise ValueError(f"Incorrect input(x) dimensions; expected 5D tensor but got shape {x.shape}")

        spatial = tuple(x.shape[-3:])  # (Z, Y, X)
        z_scaled = self.sample_latent_controlled(
            output_size=spatial,
            ctrl=ctrl,
            class_labels=class_labels,
            demographics_embedding=demographics_embedding,
            is_mod_synthetic=is_mod_synthetic,
            num_inference_steps=num_inference_steps,
        )

        # Unscale for decoding (training sampled in scaled latent space)
        z_unscaled = z_scaled / self._get_scale_factor(z_scaled)
        sample = self.decode_latent(z_unscaled)
        return {"sample": sample, "latent": z_unscaled}

    @torch.no_grad()
    def run_inference(
        self,
        x: torch.Tensor,
        class_labels: Union[torch.Tensor, int, str] = None,
        mask_one_hot: Optional[torch.Tensor] = None,
        image_masked: Optional[torch.Tensor] = None,
        controlnet_cond: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Public wrapper used by external callers (now supports control tensors)."""
        # Labels mapping (unchanged)
        labels: Optional[torch.Tensor]
        if class_labels is None or isinstance(class_labels, torch.Tensor):
            labels = class_labels
        else:
            if isinstance(class_labels, (str, int)):
                class_labels = [class_labels]
            mapped: list[int] = []
            for lab in class_labels:  # type: ignore
                if isinstance(lab, str):
                    if lab not in self.modality_map:
                        raise ValueError(f"Unknown modality {lab!r}")
                    mapped.append(self.modality_map[lab])
                else:
                    mapped.append(int(lab))
            labels = torch.tensor(mapped, dtype=torch.long, device=self.device)
        if labels is not None:
            self._validate_labels(labels)
            labels = labels.to(self.device)

        # Build control tensor
        if controlnet_cond is None:
            if mask_one_hot is None or image_masked is None:
                raise ValueError(
                    "Missing control inputs: provide either `controlnet_cond` or both `mask_one_hot` and `image_masked`."
                )
            # IMPORTANT: keep the SAME channel order [MaskedImage First, One-hot Tumor Mask]
            controlnet_cond = torch.cat(
                [image_masked.to(self.device, non_blocking=True), mask_one_hot.to(self.device, non_blocking=True)],
                dim=1,
            )
        else:
            controlnet_cond = controlnet_cond.to(self.device, non_blocking=True)

        return self._run_inference(x, class_labels=labels, ctrl=controlnet_cond, num_inference_steps=num_inference_steps)

    @torch.no_grad()
    def run_inference_bs(self, kwargs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Batch-style inference wrapper used by ``SaveMRIImages``.

        This keeps backward compatibility with the old segmentation use-case
        (mask_one_hot + image_masked), but also supports generic ControlNet
        conditioning via ``hparams.control_keys`` (e.g., modality translation).
        """

        # Reference tensor for spatial size
        ref = kwargs.get(self.target_key, None)
        
        # TODO! IMPROVE THE FOLLOWING LOGIC
        # if ref is None:
        #     # fall back to any tensor we can find (best-effort)
        #     for _k, _v in kwargs.items():
        #         if torch.is_tensor(_v) and _v.ndim == 5:
        #             ref = _v
        #             break
        if ref is None:
            raise KeyError(
                "run_inference_bs requires a 5D reference tensor under 'latent', 'target_key', or any 5D batch key."
            )

        # Build control tensor: either directly provided or assembled from keys
        if "controlnet_cond" in kwargs and torch.is_tensor(kwargs["controlnet_cond"]):
            ctrl = kwargs["controlnet_cond"].to(self.device, non_blocking=True)
        else:
            ctrl = self._prep_controlnet_cond(kwargs)

        # Labels from batch (if UNet uses class embeds)
        labels = kwargs.get("modality_map", None)
        if getattr(self.unet, "num_class_embeds", 0):
            if labels is None or not torch.is_tensor(labels):
                raise KeyError("Batch must contain 'modality_map' for class-conditioned sampling")
            labels = labels.to(self.device)

        dem_emb = self._encode_demographics(kwargs)
        is_syn = self._get_is_mod_synthetic(kwargs)

        return self._run_inference(
            x=ref,
            class_labels=labels,
            ctrl=ctrl,
            demographics_embedding=dem_emb,
            is_mod_synthetic=is_syn,
            num_inference_steps=None,
        )

