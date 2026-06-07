import contextlib
import functools
import logging
import warnings
from collections.abc import Sequence
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.networks.schedulers import DDPMScheduler, RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType

from brainmint.utils.ema import EMAConfig, ExponentialMovingAverage
from brainmint.utils.gpumem_utils import SimpleGPUMemoryTracker
from brainmint.utils.state_dict_loader import StateDictLoaderMixin

_LOG = logging.getLogger(__name__)

class DiffusionModule(StateDictLoaderMixin, pl.LightningModule):
    """
    Latent diffusion training module:
      - encodes images with a (frozen) Autoencoder to latents ``z``
      - trains a Diffusion UNet in latent space
    """

    def __init__(
        self,
        autoencoder: nn.Module | None,
        diffusion_unet: nn.Module,
        noise_scheduler: nn.Module,
        hparams: dict[str, Any],
        optimizer,     # Optimizer (Partial)
        polylr=None,   # Scheduler (Partial)
        demographics_encoder: nn.Module | None = None, 
    ):
        super().__init__()
        self.save_hyperparameters(
            hparams,
            ignore=["autoencoder", "diffusion_unet", "noise_scheduler", "optimizer", "polylr", "demographics_encoder"],
        )

        # Instantiated by Hydra
        self.autoencoder = autoencoder
        self.unet = diffusion_unet
        self.demographics_encoder = demographics_encoder 

        # ``RFlowScheduler`` inherits ``nn.Module`` but does not initialise the base
        # class, which would otherwise register it as a submodule.  Bypass
        # ``nn.Module``'s setattr logic so it is stored as a plain attribute.
        if isinstance(noise_scheduler, RFlowScheduler):
            object.__setattr__(self, "noise_scheduler", noise_scheduler)
        else:
            self.noise_scheduler = noise_scheduler


        # Factories / partials
        self.optimizer_partial = optimizer
        self.polylr_partial = polylr

        # Loss selection
        loss_name = str(self.hparams.get("loss", "L1")).upper()
        self.criterion = nn.MSELoss() if loss_name == "L2" else nn.L1Loss()

        # Hparams shortcuts
        hp = self.hparams
        latent_source = str(hp.get("latent_source", "mu")).lower()  # When Training on Outputs from VAE/AE
        if latent_source not in {"mu", "sample"}:
            raise ValueError("latent_source must be 'mu' or 'sample'")
        self.latent_source = latent_source
        self.input_key = str(hp.get("input_key", "latent"))
        self.use_modality_mapping = bool(hp.get("use_modality_mapping", True))
        self.modality_map = {str(k): int(v) for k, v in hp.get("modality_map", {}).items()}
        modality_values = list(self.modality_map.values())
        self._modality_min = min(modality_values) if modality_values else 0
        self._modality_max = max(modality_values) if modality_values else -1
        self.num_inference_steps = int(hp.get("num_inference_steps", 30))
        self.latent_channels = int(hp.get("latent_channels"))
        print(f"Latent Channels :{self.latent_channels}")

        # When an autoencoder is provided, ``encode_using_autoencoder`` controls
        # whether batches are run through it to obtain latents.  This defaults to
        # ``False`` so that code paths explicitly opt in, allowing an autoencoder
        # to be passed purely for inference-time decoding while the training data
        # already contains precomputed latents.
        self.encode_using_autoencoder = bool(hp.get("encode_using_autoencoder", False))

        # Prediction type (epsilon/sample/v_prediction) controllable via hparams
        pred_cfg = hp.get(
            "prediction_type",
            getattr(self.noise_scheduler, "prediction_type", DDPMPredictionType.EPSILON),
        )
        if isinstance(pred_cfg, DDPMPredictionType):
            self.prediction_type = pred_cfg
        else:
            s = str(pred_cfg).lower()
            if s in ("epsilon", "eps"):
                self.prediction_type = DDPMPredictionType.EPSILON
            elif s in ("sample", "z"):
                self.prediction_type = DDPMPredictionType.SAMPLE
            elif s in ("v_prediction", "v"):
                self.prediction_type = DDPMPredictionType.V_PREDICTION
            else:
                raise ValueError(f"Unknown prediction_type: {pred_cfg}")
        if hasattr(self.noise_scheduler, "prediction_type"):
            self.noise_scheduler.prediction_type = self.prediction_type
        
        # Partial posterior sampling z = μ + (mask · α · σ · ε)
        self.partial_sampling_enabled = False
        self.partial_sampling = hp.get("partial_sampling", None)
        if self.partial_sampling is not None:
            self.partial_sampling_enabled = self.partial_sampling.get("enabled", False)
            self.latent_sigma_key = str(self.partial_sampling.get("sigma_key", ""))
            if self.partial_sampling_enabled and not self.latent_sigma_key:
                raise ValueError("Partial posterior sampling enabled, yet partial_sampling.sigma_key not configured")
            print("Partial posterior sampling is enabled")


        # Runtime
        self._total_steps: int | None = None
        self._lr_scheduler = None  # populated in ``configure_optimizers`` for testing/inspection

        # Scale Factor
        sf_cfg = hp.get("scale_factor", None)
        sf_tensor: torch.Tensor | None = (
            None if (sf_cfg is None or float(sf_cfg) <= 0.0) else torch.tensor(float(sf_cfg), dtype=torch.float32)
        )
        self.register_buffer("_scale_factor", sf_tensor, persistent=True)

        self._scale_factor_batches = int(hp.get("scale_factor_batches", 1))
        self._sf_accum = None
        self._sf_count = 0
        self._sf_eps = 1e-8

        # Validation aggregation buffers (true epoch-weighted mean)
        self.register_buffer("_val_loss_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_val_count", torch.tensor(0, dtype=torch.long), persistent=False)

        # EMA validation aggregation buffers (true epoch-weighted mean)
        self.register_buffer("_val_loss_ema_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_val_ema_count", torch.tensor(0, dtype=torch.long), persistent=False)

        # Freeze AE
        if self.autoencoder is not None:
            for p in self.autoencoder.parameters():
                p.requires_grad = False
            self.autoencoder.eval()

        self.train_memory_tracker = None
        self.val_memory_tracker = None

        # Weight Decay Settings
        self.weight_decay_conf = hp.get("weight_decay", {})

        # EMA (Exponential Moving Average) of UNet weights
        ema_cfg = hp.get("ema", {}) or {}
        self._ema_enabled = bool(ema_cfg.get("enabled", False))
        self._ema_use_for_sampling = bool(ema_cfg.get("use_for_sampling", True))
        self._ema_last_step: int = -1
        self._ema_loaded: bool = False
        self.ema: ExponentialMovingAverage | None = None
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
            ema_modules = {"unet": self.unet}

            if self.demographics_encoder is not None:
                ema_modules["demographics_encoder"] = self.demographics_encoder

            # Initialize EMA over ALL selected modules
            self.ema = ExponentialMovingAverage(ema_modules, cfg=cfg)



    def _encode_image(self, images: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Encode ``images`` with the autoencoder if provided."""
        if self.autoencoder is None:
            return images, None
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=False):
                enc = getattr(self.autoencoder, "encode", None)
                if callable(enc):
                    z_mu, z_sigma = enc(images.float())
                else:
                    _, z_mu, z_sigma = self.autoencoder(images.float())
        
        return z_mu, z_sigma

    def _encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        
        z_mu, z_sigma = self._encode_image(images)

        if self.latent_source == "sample" and z_sigma is not None:
            eps = torch.randn_like(z_sigma)
            z = z_mu + z_sigma * eps
        else:
            z = z_mu
        return z

    def _get_scale_factor(self, z: torch.Tensor) -> torch.Tensor:
        """Estimate ``1/std(z)`` across the first few batches lazily from latent ``z``."""

        # If we've already computed the final scale factor and are no longer accumulating, return it.
        if self._scale_factor is not None and self._sf_accum is None:
            return self._scale_factor.to(device=z.device, dtype=z.dtype)

        # Update the running estimate and store the provisional value so that
        # training can proceed with a reasonable normalisation.
        with torch.no_grad():
            std = torch.std(z.float())
            if not torch.isfinite(std):
                std = torch.tensor(1.0, device=z.device, dtype=torch.float32)

        # Multi-batch running estimate
        self._sf_accum = std if self._sf_accum is None else (self._sf_accum + std)
        self._sf_count += 1

        mean_so_far = self._sf_accum / float(self._sf_count)
        provisional = 1.0 / (mean_so_far + self._sf_eps)

        if self._sf_count >= self._scale_factor_batches:
            final = provisional.detach()
            final = self.trainer.strategy.reduce(final, reduce_op="mean") # Lightning cross-rank average
            final = final.to(self.device)
            # Register as buffer so it moves with the module
            # self.register_buffer("_scale_factor", final, persistent=True)
            self._scale_factor = final.to(dtype=z.dtype)
            _LOG.info(f"[DiffusionModule] Estimated scale_factor={float(self._scale_factor.item()):.6f}")
            self._sf_accum = None

        return provisional
    
    # Only use it when self._scale_factor is not yet finalized.
    def _get_provisional_scale_factor(self):    
        if self._scale_factor is not None: 
            raise ValueError("Called provisional scale factor while scale factor is already finalized")

        if self._sf_accum is not None and self._sf_count > 0:
            mean_so_far = self._sf_accum / float(self._sf_count)
            provisional = 1.0 / (mean_so_far + self._sf_eps)
        else:
            provisional = torch.tensor(1., dtype=torch.float32, device=self.device)
        return provisional.to(self.device)
    

    def _target_for_prediction_type(
        self,
        pred_type: DDPMPredictionType,
        z: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        if pred_type == DDPMPredictionType.EPSILON:
            return noise

        if pred_type == DDPMPredictionType.SAMPLE:
            return z

        if pred_type == DDPMPredictionType.V_PREDICTION:
            
            # DDPM v-pred target: v = sqrt(alpha_bar_t)*eps - sqrt(1-alpha_bar_t)*x0
            if isinstance(self.noise_scheduler, DDPMScheduler):
                return self.noise_scheduler.get_velocity(sample=z, noise=noise, timesteps=timesteps)

            # For RFlow, using "velocity" target as per MAISI-MONAI 
            if isinstance(self.noise_scheduler, RFlowScheduler):
                return z - noise

            raise TypeError(f"V_PREDICTION unsupported for {type(self.noise_scheduler).__name__}")
        raise ValueError(f"Unknown prediction type: {pred_type}")


    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent -> image using AE's decode if exposed."""
        
        if self.autoencoder is None:
            raise ValueError("Autoencoder is not loaded - Cannot decode the latent image")
        else:
            if "autoencoder" not in self.get_loaded_models():
                raise ValueError("Autoencoder is not loaded - Cannot decode the latent image")

        dec = getattr(self.autoencoder, "decode", None)
        if callable(dec):
            return dec(z)
        raise ValueError("Autoencoder is missing the callable function decode")

    
    def _prep_condition_tensors(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Extract required conditioning tensors from ``batch``.
          - class_labels: from ``batch["modality_map"]`` when the UNet exposes
            ``num_class_embeds``.
        """
        if getattr(self.unet, "num_class_embeds", 0):
            if "modality_map" not in batch or not torch.is_tensor(batch["modality_map"]):
                raise ValueError("Missing 'modality_map' in batch; ensure conditioning transform is enabled")
            return {"class_labels": batch["modality_map"].to(self.device, non_blocking=True)}
        
        # Passing class labels enforced for now (will be removed)
        else:
            raise KeyError("Missing 'class_labels' for class-conditioned sampling")
            
    def _prep_demographics_condition_tensors(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Extract required Optional conditioning tensors from ``batch``.

        Optionally (for demographic conditioning):
          - demo_values:  from ``batch["demo_values"]``
          - demo_missing: from ``batch["demo_missing"]``
        """
        cond: dict[str, torch.Tensor] = {}
        
        # Demographic conditioning (Optional)
        if "demo_values" in batch:
            demo_values = batch["demo_values"]
            if not torch.is_tensor(demo_values):
                raise ValueError("Expected 'demo_values' in batch to be a torch.Tensor")
            cond["demo_values"] = demo_values.to(self.device, non_blocking=True)

        if "demo_missing" in batch:
            demo_missing = batch["demo_missing"]
            if not torch.is_tensor(demo_missing):
                raise ValueError("Expected 'demo_missing' in batch to be a torch.Tensor")
            cond["demo_missing"] = demo_missing.to(self.device, non_blocking=True)

        if self.demographics_encoder:
            if not all(key in batch for key in ["demo_values", "demo_missing"]):
                raise ValueError("Demographics Encoder Enabled, But batch missing Required Demographics Vectors")
        
        return cond


    def setup(self, stage: str | None = None) -> None:
        if hasattr(super(), "setup"):
            super().setup(stage)  # Lightning setup()

        
        # Warn if VAE is not loaded
        if self.autoencoder is None:
            warnings.warn("No autoencoder provided - latents will pass through during decode; ensure this is intentional", stacklevel=2)
        else:
            has_ae_spec = any(s.get("target") == "autoencoder" for s in self.hparams.get("weight_loads", []))
            if has_ae_spec:
                if "autoencoder" not in self.get_loaded_models():
                    raise ValueError("Autoencoder specified but weights are not loaded by StateDictLoaderMixin")
                
                
        # Compute total iterations for PolynomialLR: epochs * steps/epoch
        if stage in (None, "fit"):
            est = getattr(self.trainer, "estimated_stepping_batches", None)
            if est is None:
                train_batches = len(self.trainer.datamodule.train_dataloader())
                est = self.trainer.max_epochs * train_batches
            self._total_steps = int(est)
        
        # If EMA is enabled and we did not restore EMA from a checkpoint, reset EMA to
        # match the (possibly externally loaded) UNet weights.
        if self._ema_enabled and self.ema is not None and (not self._ema_loaded) and stage in (None, "fit"):
            self.ema.reset()
        
    def on_fit_start(self) -> None:
        if self.train_memory_tracker is None:
            self.train_memory_tracker = SimpleGPUMemoryTracker(device=self.device)
        if self.val_memory_tracker is None:
            self.val_memory_tracker = SimpleGPUMemoryTracker(device=self.device)
        
        # Move EMA shadow to the right device (unless configured to store on CPU).
        if self._ema_enabled and self.ema is not None:
            self.ema.to(self.device)
    
    @contextlib.contextmanager
    def ema_scope(self, enabled: bool = True):
        """Temporarily swap UNet params to EMA weights within the scope."""
        if not (enabled and self._ema_enabled and self.ema is not None):
            yield
            return
        self.ema.store()  # store all tracked modules
        self.ema.copy_to()  # copy EMA weights to all tracked modules
        try:
            yield
        finally:
            self.ema.restore()  # restore original weights for all tracked modules

    def _get_weight_decay_from_partial(self) -> float:
        # Hydra _partial_ gives functools.partial
        kw = getattr(self.optimizer_partial, "keywords", {}) or {}
        wd = kw.get("weight_decay", 0.0)
        try:
            return float(wd)
        except Exception:
            return 0.0

    def _assert_weight_decay_config(self):
        
        wd_enabled = bool(self.weight_decay_conf.get("enabled", False))
        #split_groups = bool(self.weight_decay_conf.get("split_param_groups", True))
        opt_wd = self._get_weight_decay_from_partial()
        opt_tar = self._get_opt_target_name()

        if wd_enabled:
            wrong_optim = ( opt_tar != "torch.optim.adamw.AdamW" and not opt_tar.endswith(".AdamW") )
            if wrong_optim or opt_wd <= 0.0:
                raise ValueError("Optimizer is configured incorrectly!") 


    def _get_opt_target_name(self) -> str:
        """Best-effort name of the optimizer class/function used by optimizer_partial."""
        opt = getattr(self, "optimizer_partial", None)
        if isinstance(opt, functools.partial):
            func = opt.func
            return f"{getattr(func, '__module__', '')}.{getattr(func, '__name__', func.__class__.__name__)}"
        # fallback
        return str(type(opt))

    def _build_param_groups(self, modules: list[tuple[str, nn.Module]], weight_decay: float):
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

    def configure_optimizers(self):
        
        modules = [("unet", self.unet)]
        if self.demographics_encoder is not None:
            modules.append(("demographics_encoder", self.demographics_encoder))

        wd_enabled = bool(self.weight_decay_conf.get("enabled", False))
        split_groups = bool(self.weight_decay_conf.get("split_param_groups", True))
        wd = self._get_weight_decay_from_partial()

        if wd_enabled and split_groups and wd > 0:
            self._assert_weight_decay_config()    
            param_groups = self._build_param_groups(modules, wd)
            opt = self.optimizer_partial(params=param_groups)
        else:

            # No weight decay configured -> just pass params
            # Collect parameters from UNet (+ Demographics encoder, if present)
            params = list(self.unet.parameters())
            if self.demographics_encoder is not None:
                params += list(self.demographics_encoder.parameters())
            opt = self.optimizer_partial(params=params)

        self._log_learning_rates("configured", opt)
        if self.polylr_partial is not None:
            if self._total_steps is None:
                raise RuntimeError("total steps not computed; check setup('fit').")
            sched = self.polylr_partial(optimizer=opt, total_iters=self._total_steps)
            self._lr_scheduler = sched
            # Lightning will step this scheduler automatically every batch (interval="step")
            return {
                "optimizer": opt,
                 "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
            }
        return opt

    def _tb_writer(self):
        logs = self.trainer.loggers if getattr(self.trainer, "loggers", None) else (
            [self.trainer.logger] if getattr(self.trainer, "logger", None) else []
        )
        for lg in logs:
            if isinstance(lg, pl.loggers.TensorBoardLogger):
                return lg.experiment
        return None
        
    def _compute_z(self, batch: dict[str, Any], partial_sampling: bool = False) -> torch.Tensor:
        """Extract and optionally encode latents from ``batch``.

        When ``encode_using_autoencoder`` is ``True`` the batch is expected to
        contain raw images under :attr:`input_key` and an autoencoder must be
        supplied.  Otherwise the method treats the tensor at ``input_key`` as
        precomputed latents.  In both cases the scale factor is applied (and
        estimated on the fly if needed).
        """
        
        # Note, x can be image as well!
        x = batch[self.input_key].to(self.device, non_blocking=True)
        
        z_mu = None
        z_sigma = None
        
        if self.encode_using_autoencoder:
            if self.autoencoder is None:
                raise ValueError("encode_using_autoencoder=True but no autoencoder provided")
            if self.input_key != "image":
                raise ValueError("encode_using_autoencoder expects input_key='image'")
            with torch.no_grad():
                z_mu, z_sigma = self._encode_image(x)
        else:
            z_mu = x
            if self.partial_sampling_enabled and partial_sampling:
                if self.latent_sigma_key not in batch:
                    raise KeyError(f"Partial sampling requested but batch missing '{self.latent_sigma_key}'")
                z_sigma = batch[self.latent_sigma_key].to(self.device, non_blocking=True)
        
        if partial_sampling and self.partial_sampling_enabled:

            eps = torch.randn_like(z_mu)

            p = float(self.partial_sampling.get("sigma_prob", 0.0))
            alpha = float(self.partial_sampling.get("sigma_alpha", 0.0))
                
            # per-sample mask (broadcast over C,Z,Y,X)
            mask = (torch.rand(z_mu.size(0), 1, 1, 1, 1, device=z_mu.device) < p).float()
                
            z = z_mu + mask * (alpha * z_sigma * eps)
        
        else:
            z = z_mu

        return z * self._get_scale_factor(z)

    def _apply_noise(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random noise, timesteps and perturb latent ``z``."""
        noise = torch.randn_like(z)
        if isinstance(self.noise_scheduler, RFlowScheduler):
            timesteps = self.noise_scheduler.sample_timesteps(z)
        else:
            num_tr = int(self.noise_scheduler.num_train_timesteps)
            timesteps = torch.randint(0, num_tr, (z.shape[0],), device=z.device).long()

        z_noisy = self.noise_scheduler.add_noise(original_samples=z, noise=noise, timesteps=timesteps)
        return timesteps, noise, z_noisy

    def _encode_demographics(self, batch: dict[str, Any]) -> torch.Tensor | None:
        """
        Turn raw demographics from the batch into an embedding vector.

        Expects the batch to provide:
        -----------------------------
        * ``demo_values``:  (B, D) tensor with dense demographic features.
        * ``demo_missing``: (B, D) tensor mask aligned with ``demo_values``.

        Returns
        -------
        * ``demographics_embedding``: (B, D_dem) tensor to pass to the UNet, or
          ``None`` if the UNet is not configured with demographics.

        Raises
        ------
        * ValueError / TypeError with clear messages if the configuration is
          inconsistent (e.g. UNet expects demographics but the batch or encoder
          is missing).
        """
        # If the UNet doesn't use demographics, just ignore any demo data.
        if not getattr(self.unet, "with_demographics", False):
            return None

        # UNet expects demographics, so we must have an encoder.
        if self.demographics_encoder is None:
            raise RuntimeError(
                "DiffusionUNet was created with with_demographics=True but "
                "DiffusionModule.demographics_encoder is None. Either disable "
                "with_demographics in the UNet config or provide a demographics encoder."
            )

        has_demo_values = "demo_values" in batch
        has_demo_missing = "demo_missing" in batch

        if not (has_demo_values and has_demo_missing):
            raise ValueError(
                "Diffusion UNet is demographics-conditioned (with_demographics=True) "
                "but batch does not provide both 'demo_values' and 'demo_missing'."
            )

        demo_values = batch["demo_values"]
        demo_missing = batch["demo_missing"]

        if not torch.is_tensor(demo_values) or not torch.is_tensor(demo_missing):
            raise TypeError("'demo_values' and 'demo_missing' must both be tensors")

        if demo_values.shape != demo_missing.shape:
            raise ValueError(
                f"'demo_values' and 'demo_missing' must have the same shape, "
                f"got {tuple(demo_values.shape)} vs {tuple(demo_missing.shape)}"
            )

        demo_values = demo_values.to(self.device, non_blocking=True)
        demo_missing = demo_missing.to(self.device, non_blocking=True)

        # Black-box encoder: (B, D) + (B, D) -> (B, D_dem)
        dem_emb = self.demographics_encoder(demo_values, demo_missing)
        if not torch.is_tensor(dem_emb):
            raise TypeError(
                "demographics_encoder must return a tensor, "
                f"got {type(dem_emb).__name__}"
            )

        if dem_emb.ndim != 2:
            raise ValueError(
                "demographics_encoder must return a (B, D_dem) tensor, "
                f"got shape {tuple(dem_emb.shape)}"
            )

        if dem_emb.shape[0] != demo_values.shape[0]:
            raise ValueError(
                "Batch size mismatch between demographics embedding and inputs: "
                f"{dem_emb.shape[0]} vs {demo_values.shape[0]}"
            )

        return dem_emb


    def _log_learning_rates(self, when: str, opt: torch.optim.Optimizer | None = None) -> None:
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
            _LOG.info(f"[DiffusionModule] Learning rate ({when}): {' | '.join(lr_msgs)}")

    def on_train_epoch_start(self) -> None:
        if self.train_memory_tracker is not None:
            self.train_memory_tracker.reset_peak()
        self._log_learning_rates(f"start epoch {self.current_epoch}")


    def training_step(self, batch: dict[str, Any], batch_idx: int):

        z = self._compute_z(batch, partial_sampling=True)
        bs = z.size(0)
        timesteps, noise, z_noisy = self._apply_noise(z)

        unet_inputs = {"x": z_noisy, "timesteps": timesteps}
        unet_inputs.update(self._prep_condition_tensors(batch))
        
        dem_cond_inputs = self._prep_demographics_condition_tensors(batch)
        dem_emb = self._encode_demographics(dem_cond_inputs)
        if dem_emb is not None:
            unet_inputs["demographics_embedding"] = dem_emb
        
        pred = self.unet(**unet_inputs)

        target = self._target_for_prediction_type(
                pred_type=self.prediction_type, 
                z=z, 
                noise=noise, 
                timesteps=timesteps
            )

        loss = self.criterion(pred.float(), target.float())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs, logger=False)
        
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

    def on_validation_epoch_start(self) -> None:
        if self.val_memory_tracker is not None:
            self.val_memory_tracker.reset_peak()

    def validation_step(self, batch: dict[str, Any], batch_idx: int):

        z = self._compute_z(batch, partial_sampling=False)
        bs = z.size(0)
        timesteps, noise, z_noisy = self._apply_noise(z)

        base_inputs = {"x": z_noisy, "timesteps": timesteps}
        base_inputs.update(self._prep_condition_tensors(batch))
        
        unet_inputs = dict(base_inputs)
        dem_cond_inputs = self._prep_demographics_condition_tensors(batch)
        dem_emb = self._encode_demographics(dem_cond_inputs)
        if dem_emb is not None:
            unet_inputs["demographics_embedding"] = dem_emb

        pred = self.unet(**unet_inputs)

        target = self._target_for_prediction_type(
                pred_type=self.prediction_type, 
                z=z, 
                noise=noise, 
                timesteps=timesteps
            )

        loss = self.criterion(pred.float(), target.float())
        
        # accumulate for true epoch-weighted average
        self._val_loss_sum += loss.detach() * bs
        self._val_count += bs

        # Compute EMA val loss (extra forward pass, no gradients)
        if self._ema_enabled and (self.ema is not None):
            with self.ema_scope(enabled=True):
                
                unet_inputs_ema = dict(base_inputs)
                dem_emb_ema = self._encode_demographics(dem_cond_inputs)
                if dem_emb_ema is not None:
                    unet_inputs_ema["demographics_embedding"] = dem_emb_ema
                
                pred_ema = self.unet(**unet_inputs_ema)

            loss_ema = self.criterion(pred_ema.float(), target.float())
            self._val_loss_ema_sum += loss_ema.detach() * bs
            self._val_ema_count += bs

        return loss

    def on_validation_epoch_end(self):
        # Global (across ranks) weighted mean: sum(loss * n) / sum(n)
        sum_all = self.trainer.strategy.reduce(self._val_loss_sum, reduce_op="sum")
        cnt_all = self.trainer.strategy.reduce(self._val_count, reduce_op="sum")
        val_loss = sum_all / cnt_all.clamp_min(1).float()

        # log once per epoch; already globally reduced so no need for sync_dist here
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=False)
        self.log("val_total", val_loss, prog_bar=False, logger=False, sync_dist=False) # alias for filename/monitor

        # EMA val loss
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

        if self.val_memory_tracker is not None:
            _LOG.info(self.val_memory_tracker.record_peak_memory(self.current_epoch, "VAL"))

    def on_train_epoch_end(self) -> None:
        if self.train_memory_tracker is not None:
            _LOG.info(self.train_memory_tracker.record_peak_memory(self.current_epoch, "TRAIN"))
        #self._log_learning_rates(f"end epoch {self.current_epoch}")

        avg_loss = self.trainer.callback_metrics.get("train/loss_epoch")
        if avg_loss is not None:
            self.log("train/loss_epoch_avg", avg_loss, prog_bar=True, logger=True)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA after optimizer step (global_step only increments when an optimizer step occurs).
        if not (self._ema_enabled and self.ema is not None):
            return
        step = int(self.global_step)
        if step <= 0:
            return
        if step == self._ema_last_step:
            return
        self.ema.update(step=step)  # updates all tracked modules
        self._ema_last_step = step

    
    def on_save_checkpoint(self, checkpoint):
        if self._scale_factor is not None:
            checkpoint["scale_factor"] = self._scale_factor.detach().float().cpu()

        if self._ema_enabled and self.ema is not None:
            # convenience: top-level nested dict for StateDictLoaderMixin paths
            checkpoint["ema_state_dict"] = self.ema.ema_state_dict()
            checkpoint["ema_last_step"] = int(self._ema_last_step)

    def on_load_checkpoint(self, checkpoint):
        sf = checkpoint.get("scale_factor", None)
        if sf is not None:
            self._scale_factor = sf
            #self.register_buffer("_scale_factor", sf.to(self.device), persistent=True)
            _LOG.info(f"[DiffusionModule] Restored scale_factor={self._scale_factor.item():.6f}")

        if not (self._ema_enabled and self.ema is not None):
            return

        # Preferred: restore full EMA tracker state
        ema_state = checkpoint.get("ema", None)

        # Fallback: if only ema_state_dict was saved
        if ema_state is None:
            ema_sd = checkpoint.get("ema_state_dict", None)
            if ema_sd is not None:
                ema_state = {"ema_state_dict": ema_sd}

        if ema_state is not None:
            try:
                self.ema.load_state_dict(ema_state)
                self._ema_last_step = int(checkpoint.get("ema_last_step", -1))
                self._ema_loaded = True
                _LOG.info("[DiffusionModule] Restored EMA state.")
            except Exception as e:
                _LOG.warning("[DiffusionModule] Failed to restore EMA state: %s", e)


    @torch.no_grad()
    def sample_latent(
        self,
        output_size: tuple[int, int, int], # Must be equal to input size Z, Y, X
        batch_size: int = 1,
        class_labels: torch.Tensor | None = None,
        demo_values: torch.Tensor | None = None,
        demo_missing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate latent with scheduler loop; returns latent tensor ``(B, C, Z, Y, X)``."""

        img = torch.randn((batch_size, self.latent_channels, *output_size), device=self.device)

        # Set timesteps for inference
        if isinstance(self.noise_scheduler, RFlowScheduler):
            self.noise_scheduler.set_timesteps(
                device=self.device,
                num_inference_steps=self.num_inference_steps,
                input_img_size_numel=img[0].numel(),
            )
        else:
            self.noise_scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
            if isinstance(self.noise_scheduler, DDPMScheduler) and self.num_inference_steps < self.noise_scheduler.num_train_timesteps:
                raise ValueError(
                    "* WARNING: Image noise_scheduler is a DDPMScheduler and num_inference_steps: "
                    f"{self.num_inference_steps}\n"
                    "* Set num_inference_steps = noise_scheduler.num_train_timesteps = "
                    f"{self.noise_scheduler.num_train_timesteps}"
                )

        all_t = self.noise_scheduler.timesteps.to(self.device)

        with self.ema_scope(enabled=self._ema_use_for_sampling):
            
            # Compute demographics embedding 
            if demo_values is not None or demo_missing is not None:
                emb_encoder_inp: dict[str, Any] = {}
                if demo_values is not None:
                    emb_encoder_inp["demo_values"] = demo_values
                if demo_missing is not None:
                    emb_encoder_inp["demo_missing"] = demo_missing
                dem_emb = self._encode_demographics(emb_encoder_inp)
            else:
                dem_emb = None
            

            for i, t in enumerate(all_t):
                t_val = t.item()
                next_t_val = all_t[i + 1].item() if i + 1 < len(all_t) else 0
                t_batch = torch.full((batch_size,), t_val, device=self.device, dtype=all_t.dtype)
                inputs: dict[str, Any] = {"x": img, "timesteps": t_batch}
                if getattr(self.unet, "num_class_embeds", 0):
                    if class_labels is None:
                        raise ValueError("Missing 'class_labels' for class-conditioned sampling")
                    if class_labels.numel() != batch_size:
                        raise ValueError(f"Value Mismatch, '# of class_labels: {class_labels.numel()}' must match batchsize: {batch_size}")
                    inputs["class_labels"] = class_labels.to(self.device)
                
                # Demographics Conditioning
                if dem_emb is not None:
                    inputs["demographics_embedding"] = dem_emb

                # Just enforcing class labels (Can be removed)
                if getattr(self.unet, "num_class_embeds", 0) and "class_labels" not in inputs:
                    raise KeyError("Missing 'class_labels' for class-conditioned sampling")

                pred = self.unet(**inputs)
                pred = pred.to(dtype=img.dtype)
                if isinstance(self.noise_scheduler, RFlowScheduler):
                    img, _ = self.noise_scheduler.step(pred, t_val, img, next_t_val)
                else:
                    img, _ = self.noise_scheduler.step(pred, t_val, img)  # DDPM

        return img  # latent


    @torch.no_grad()
    def sample(
        self,
        output_size: tuple[int, int, int],
        batch_size: int = 1,
        class_labels: torch.Tensor | None = None,
        demo_values: torch.Tensor | None = None,
        demo_missing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate decoded images using :meth:`sample_latent` and the autoencoder."""
        z = self.sample_latent(
            output_size, 
            batch_size=batch_size, 
            class_labels=class_labels,
            demo_values=demo_values,
            demo_missing=demo_missing,
        )
        sf = self._scale_factor if self._scale_factor is not None else self._get_provisional_scale_factor()
        z_unscaled = z / sf.to(z.device)
        return self.decode_latent(z_unscaled)

    @torch.no_grad()
    def _run_inference(
        self,
        x: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        demo_values: torch.Tensor | None = None,
        demo_missing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Helper for :class:`SaveMRIImages` callback.

        Generates a fresh latent using :meth:`sample_latent` that matches the
        spatial size of ``x`` and, when an autoencoder is available, decodes it
        back to image space.  Both the decoded sample and the latent are
        returned so the callback can persist them separately.
        """
        if len(x.shape) != 5:
            raise ValueError( f"Incorrect input(x) Dimensions, shape:{x.shape}, Length of Dim must be == 5")
        
        spatial = x.shape[-3:]
        b = x.shape[0]

        z = self.sample_latent(
            spatial, 
            batch_size=b, 
            class_labels=class_labels,
            demo_values=demo_values,
            demo_missing=demo_missing,
        )
        sf = self._scale_factor if self._scale_factor is not None else self._get_provisional_scale_factor()
        z_unscaled = z / sf.to(z.device)
        sample = self.decode_latent(z_unscaled)
        return {"sample": sample, "latent": z_unscaled}

    def _validate_labels(self, labels: torch.Tensor) -> None:
        if labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("class_labels tensor must contain integers")
        if (labels < self._modality_min).any() or (labels > self._modality_max).any():
            raise ValueError(
                f"Invalid class label(s) {labels.tolist()}; valid range: [{self._modality_min}, {self._modality_max}]"
            )
    
    @torch.no_grad()
    def run_inference(
        self,
        x: torch.Tensor,
        class_labels: torch.Tensor | int | str | Sequence[int | str] | None = None,
        demo_values: torch.Tensor | None = None,
        demo_missing: torch.Tensor | None = None,
    ):
        """Public wrapper used by external callers and callbacks.

        ``class_labels`` may be provided as a tensor of integer indices, a
        single int/str, or a sequence of int/str.  String labels are mapped via
        :attr:`modality_map` when present.
        """

        labels: torch.Tensor | None
        if class_labels is None or isinstance(class_labels, torch.Tensor):
            labels = class_labels  # may be None
        else:
            if isinstance(class_labels, (str, int)):
                class_labels = [class_labels]
            mapped: list[int] = []
            for lab in class_labels:  # type: ignore[iteration-over-optional]
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
        return self._run_inference(x, class_labels=labels, demo_values=demo_values, demo_missing=demo_missing)

    @torch.no_grad()
    def run_inference_from_image(
        self,
        image: torch.Tensor,
        class_labels: torch.Tensor | int | str | Sequence[int | str] | None = None,
        demo_values: torch.Tensor | None = None,
        demo_missing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Convenience wrapper:
        image -> (autoencoder.encode) -> latent z
        z -> scale_factor -> z_scaled
        then calls self.run_inference(x=z_scaled, ...)
        Returns the same dict as run_inference(): {"sample": ..., "latent": ...}
        """
        if image.ndim != 5:
            raise ValueError(f"Expected image to be 5D (B,C,Z,Y,X), got shape={tuple(image.shape)}")

        if self.autoencoder is None or "autoencoder" not in self.get_loaded_models():
            raise ValueError("Autoencoder is not loaded - cannot run image->latent inference")

        image = image.to(self.device, non_blocking=True)

        # Encode -> latent (mu or sample depending on self.latent_source)
        z = self._encode_to_latent(image)  # unscaled latent

        # Use the SAME scale factor logic as training (prefer finalized factor; else provisional; else per-batch fallback)
        if self._scale_factor is not None:
            sf = self._scale_factor.to(device=z.device, dtype=z.dtype)
        elif self._sf_accum is not None and self._sf_count > 0:
            # Provisional estimate exists
            sf = self._get_provisional_scale_factor().to(device=z.device, dtype=z.dtype)
        else:
            # fallback (early sanity check): compute from current z
            std = torch.std(z.float())
            if not torch.isfinite(std) or std <= 0:
                std = torch.tensor(1.0, device=z.device, dtype=torch.float32)
            sf = (1.0 / (std + self._sf_eps)).to(device=z.device, dtype=z.dtype)

        z_scaled = z * sf

        return self.run_inference(
            x=z_scaled,
            class_labels=class_labels,
            demo_values=demo_values,
            demo_missing=demo_missing,
        )
    @torch.no_grad()
    
    def run_inference_image_bs(self, kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Dict-style wrapper for callbacks that call infer_fn(infer_kwargs).

        Expected kwargs keys:
        - "image": Tensor (B,C,Z,Y,X)
        - "modality_map" (optional): Tensor (B,) class labels (your convention)
        - "demo_values"/"demo_missing" (optional)
        """
        if "image" not in kwargs:
            raise KeyError(f"run_inference_image_bs expected key 'image'. Got keys={list(kwargs.keys())}")

        image = kwargs["image"]

        # Prefer your standard batch key "modality_map" for class conditioning
        class_labels = kwargs.get("modality_map", None)

        demo_values = kwargs.get("demo_values", None)
        demo_missing = kwargs.get("demo_missing", None)

        return self.run_inference_from_image(
            image=image,
            class_labels=class_labels,
            demo_values=demo_values,
            demo_missing=demo_missing,
        )
    
    def run_inference_bs(self, kwargs: dict[str, torch.Tensor]):
        """BrainScape-style wrapper that accepts a batch dict.

        Expected keys:
          • 'latent': input latent tensor
          • 'modality_map': class label tensor
          • 'demo_values' (Optional): demo cond tensor
          • 'demo_missing' (Optional): demo cond tensor
        """
        latent = kwargs['latent']
        labels = kwargs['modality_map']
        demo_values = kwargs.get("demo_values", None)
        demo_missing = kwargs.get("demo_missing", None)

        ## TODO: REMOVE IT - TEMP USE ONLY
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # If UNet requires demographics but they weren't provided, create dummy "all N/A"
        if getattr(self.unet, "with_demographics", False) and (demo_values is None or demo_missing is None):
            b = int(latent.shape[0])
            demo_values, demo_missing = self._make_dummy_demographics(
                batch_size=b, device=self.device, dtype=torch.float32
            )
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        return self.run_inference(x=latent, class_labels=labels, demo_values=demo_values, demo_missing=demo_missing)


    ## TODO: REMOVE IT - TEMP USE ONLY
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _infer_demo_feature_dim(self) -> int:
        """
        Best-effort inference of the demographics feature dimension D.
        Falls back to a clear error if we can't infer it.
        """
        enc = self.demographics_encoder
        if enc is None:
            raise RuntimeError("Cannot infer demo feature dim: demographics_encoder is None")

        # Common patterns: encoder has .config (dict-like) with ordered_fields
        cfg = getattr(enc, "config", None)
        if isinstance(cfg, dict):
            of = cfg.get("ordered_fields", None)
            if isinstance(of, (list, tuple)) and len(of) > 0:
                return len(of)

        # Sometimes stored as attribute, e.g. .ordered_fields
        of = getattr(enc, "ordered_fields", None)
        if isinstance(of, (list, tuple)) and len(of) > 0:
            return len(of)

        raise RuntimeError(
            "Could not infer demographics feature dimension D. "
            "Please expose encoder.config['ordered_fields'] (or encoder.ordered_fields), "
            "or pass demo_values/demo_missing explicitly."
        )    
    def _make_dummy_demographics(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create dummy demographics corresponding to all 'N/A'.

        demo_values: zeros  (safe placeholder; categorical 'n/a' is usually 0)
        demo_missing: ones (marks every field as missing)
        """
        d = self._infer_demo_feature_dim()
        demo_values = torch.zeros((batch_size, d), device=device, dtype=dtype)
        demo_missing = torch.ones((batch_size, d), device=device, dtype=dtype)
        return demo_values, demo_missing

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




    # @torch.no_grad()
    # def run_inference_ldm_bs(self, infer_kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    #     """
    #     MONAI UKB-LDM inference (matches LDM UKB Trained Model):
    #     - latent noise shape: (B, 3, 20, 28, 20) for output 160x224x160
    #     - concat cond channels (4) => in_channels=7
    #     - context conditioning shape (B,1,4)
    #     - DDIMScheduler loop
    #     """
    #     lat = infer_kwargs.get("latent", None)
    #     if lat is None or not torch.is_tensor(lat) or lat.ndim != 5:
    #         raise KeyError("run_inference_ldm_bs expects a 5D tensor under infer_kwargs['latent'].")

    #     device = lat.device
    #     b = int(lat.shape[0])
    #     latent_spatial = tuple(int(x) for x in lat.shape[-3:])  # e.g. (20, 28, 20)

    #     cond_vals = list(self.hparams.get("ldm_conditioning_values", [0.0, 0.1, 0.2, 0.4]))
    #     if not isinstance(cond_vals, (list, tuple)) or len(cond_vals) != 4:
    #         raise ValueError("hparams.ldm_conditioning_values must be a list/tuple of length 4.")

    #     cond = torch.tensor(cond_vals, device=device, dtype=torch.float32).view(1, 4).repeat(b, 1)  # (B,4)
    #     context = cond.unsqueeze(1)  # (B,1,4)

    #     cond_concat = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)      # (B,4,1,1,1)
    #     cond_concat = cond_concat.expand(b, 4, *latent_spatial)           # (B,4,20,28,20)

    #     # Start from pure noise with EXACT same shape as provided latent
    #     img = torch.randn_like(lat, dtype=torch.float32)

    #     self.noise_scheduler.set_timesteps(num_inference_steps=int(self.num_inference_steps))
    #     timesteps = self.noise_scheduler.timesteps.to(device)

    #     for t in timesteps:
    #         t_batch = torch.full((b,), int(t.item()), device=device, dtype=torch.long)
    #         model_in = torch.cat((img, cond_concat), dim=1)  # (B, latent_ch+4, ...)

    #         pred = self.unet(x=model_in, timesteps=t_batch, context=context).to(dtype=img.dtype)

    #         try:
    #             img, _ = self.noise_scheduler.step(pred, t, img)
    #         except Exception:
    #             img, _ = self.noise_scheduler.step(pred, int(t.item()), img)

    #     sample = self.decode_latent(img.float())
    #     return {"sample": sample, "latent": img}