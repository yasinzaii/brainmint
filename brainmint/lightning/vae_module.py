import logging
from typing import Dict, List, Tuple, Any, Optional

import torch
import pytorch_lightning as pl
from hydra.utils import instantiate

from monai.inferers import SlidingWindowInferer, SimpleInferer
from brainmint.utils.gpumem_utils import SimpleGPUMemoryTracker
from brainmint.inference.dynamic_inference import dynamic_infer
from brainmint.utils.state_dict_loader import StateDictLoaderMixin

_LOG = logging.getLogger(__name__)

class VAEModule(StateDictLoaderMixin, pl.LightningModule):
    
    def __init__(
        self,
        loss_manager,   # VaeLossManager (Partial)
        autoencoder,    # Instantiated by Hydra
        discriminator,  # Instantiated by Hydra
        inferer,        # Instantiated by Hydra
        hparams,        # lr, weights, sched list, eps dict
        optimizer,      # Optimizer (Partial)
        scheduler,      # Scheduler (Partial)

        # Seperate Schedulers For GAN's D & G
        scheduler_g=None,  # Optional generator scheduler (Partial)
        scheduler_d=None,  # Optional discriminator scheduler (Partial)
    ):
        super().__init__()
        self.save_hyperparameters(
            hparams,
            ignore=[
                "autoencoder",
                "discriminator",
                "inferer",
                "optimizer",
                "scheduler",
                "scheduler_g",
                "scheduler_d",
            ],
        )

        self.automatic_optimization = False  # * Manual opt for GANs

        # Instantiated by Hydra
        self.inferer = inferer
        self.autoencoder = autoencoder
        self.discriminator = discriminator

        # Factories / partials
        self.optimizer_partial = optimizer
        self.loss_manager_partial = loss_manager

        self.scheduler_partial = scheduler
        self.scheduler_g_partial = scheduler_g or scheduler
        self.scheduler_d_partial = scheduler_d or scheduler
        
        # Bookkeeping
        self.best_val_loss = float("inf")
        self.val_interval = 10  # log sample images every N val epochs
        # self.save_hyperparameters(
        #     {"lr": hparams["lr"], "lambdaLR": hparams["lambdaLR"], "adam_eps": hparams["adam_eps"]},
        #     ignore=["loss_manager", "optimizer", "scheduler"],  # large objects
        # )

        self.loss_manager = None
        self.memory_tracker = None

        self.learning_rate = self.hparams["lr"]

        
    def setup(self, stage: Optional[str] = None) -> None:
        if hasattr(super(), "setup"):
            super().setup(stage)  # Lightning setup()

        # Finalize loss manager (was partial)
        self.loss_manager = self.loss_manager_partial(device=self.device) 
        self.memory_tracker = SimpleGPUMemoryTracker(device=self.device)

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon, z_mu, z_sigma = self.autoencoder(x)
        return recon, z_mu, z_sigma

    @torch.no_grad()
    def _run_inference(self, x: torch.Tensor) -> torch.Tensor:
        recon, z_mu, z_sigma = dynamic_infer(inferer=self.inferer, model=self.autoencoder, images=x)
        return recon, z_mu, z_sigma

    def run_inference(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Proxy for callbacks expecting a batch dict.
        Expects batch['image'] and ignores extra keys.
        """
        image = batch['image']
        return self._run_inference(x=image)
        

    def on_train_epoch_start(self) -> None:
        self.loss_manager.set_mode("train")  
        self.loss_manager.reset_epoch("train")
        self.memory_tracker.reset_peak()
        
        #TODO - MAYBE LOG EPOC START DETAILS 


    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        images = batch["image"].contiguous()
        bs = images.size(0)

        # Get optimizers (Manual opt)
        optims = self.optimizers()
        if isinstance(optims, (list, tuple)):
            opt_g = optims[0]; opt_d = optims[1] if len(optims) > 1 else None
        else:
            opt_g = optims; opt_d = None
            
        has_disc = self.discriminator is not None and self.loss_manager.adv_w > 0 

        # Generator Pass
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = self.autoencoder(images)
        logits_fake = self.discriminator(reconstruction.contiguous().float())[-1] if has_disc else torch.zeros(1, device=reconstruction.device)
        loss_g = self.loss_manager.gen_loss(images=images, recon=reconstruction,
                    z_mu=z_mu, z_sigma=z_sigma, logits_fake=logits_fake, track_stats=True, bs=bs)

        self.manual_backward(loss_g)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # Discriminator Pass
        if has_disc and (opt_d is not None):
            self.toggle_optimizer(opt_d)
            opt_d.zero_grad(set_to_none=True)
            logits_real = self.discriminator(images.contiguous())[-1]
            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]

            loss_d = self.loss_manager.disc_loss(logits_real=logits_real,
                            logits_fake=logits_fake, bs=bs, track_stats=True)

            self.manual_backward(loss_d)
            opt_d.step()
            self.untoggle_optimizer(opt_d)
        
        # Increment the sample counts by batch size
        self.loss_manager.inc_samples_count(bs=bs)  
    
        # progress bar only (no TB/CSV)
        self.log("train/total", float(loss_g), on_step=True, on_epoch=False, prog_bar=True, logger=False) 
        # # TB-only step logging every "Trainer(log_every_n_steps=...)"
        metrics = {k: (v.detach().item() if torch.is_tensor(v) else float(v)) for k, v in self.loss_manager.last_loss["train"].items()}
        log_every_n_steps = int(getattr(self.trainer, "log_every_n_steps", 100))
        tb_writer = self._tb_writer()
        step_ix = int(self.global_step) + 1
        if (log_every_n_steps > 0) and (step_ix % log_every_n_steps == 0) and self.trainer.is_global_zero and tb_writer is not None:
             for k, v in metrics.items():
                tb_writer.add_scalar(f"train/{k}", v, step_ix)
             tb_writer.add_scalar("train/total", float(loss_g), step_ix)
    

    def on_validation_epoch_start(self) -> None:
        self.loss_manager.set_mode("val")
        self.loss_manager.reset_epoch("val")
        self.memory_tracker.reset_peak()

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        bs = images.size(0)

        reconstruction, z_mu, z_sigma = self._run_inference(images)
        val_loss = self.loss_manager.val_loss(images=images, recon= reconstruction,
                            z_mu=z_mu, z_sigma=z_sigma, bs=bs, track_stats=True)

        self.loss_manager.inc_samples_count(bs=bs)  # Increment sample counter by batch size


    def on_validation_epoch_end(self) -> None:
        epoch_avgs = self.loss_manager.epoch_averages("val")
        total_avg  = self.loss_manager.weighted_sum_skip_kl(epoch_avgs) # Recon + Precep
        self.loss_manager.store_epoch("val")
        

        self.log_dict({f"val_epoch_avg/{k}": float(v) for k, v in epoch_avgs.items()}, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_epoch_avg/total", float(total_avg), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_total", float(total_avg), on_step=False, on_epoch=True, logger=False)  # alias for filename/monitor

        _LOG.info(self.memory_tracker.record_peak_memory(self.current_epoch, "VAL"))

        # TODO
        # scale_factor_sample = 1.0 / z_mu.flatten().std()
        #     tensorboard_writer.add_scalar("val_one_sample_scale_factor", scale_factor_sample, epoch)
        # Maye write a speific callback for logging scale factor - as it requires access to zmu and sigma.

        # TODO - LOG THE IMAGES TO TENSOR BOARD SUMMARY WRITER 

    def on_train_epoch_end(self) -> None:
        self.loss_manager.check_consistency() # simple count check
        epoch_avgs = self.loss_manager.epoch_averages("train") # Avg epoch training loss
        total_avg  = self.loss_manager.weighted_sum(epoch_avgs) # RECON + KL + PREC

        # Step LR Schedulers **per-epoch** (as lambda is epoch-based)
        scheds = self.lr_schedulers()
        if isinstance(scheds, (list, tuple)):
            for sch in scheds:
                sch.step()
        elif scheds is not None:
            scheds.step()

        self.log_dict({f"train_epoch_avg/{k}": float(v) for k, v in epoch_avgs.items()}, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_epoch_avg/total", float(total_avg), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        self.loss_manager.store_epoch("train") # Finally store epoch record and clear running state
        _LOG.info(self.memory_tracker.record_peak_memory(self.current_epoch, "TRAIN"))
        
        # Log Optimizer G Learning Rate 
        optims = self.optimizers()
        if isinstance(optims, (list, tuple)):
            g_lr = optims[0].param_groups[0]["lr"]  
            self.log("lr/g", g_lr, on_epoch=True, logger=False)  # tracked by Trainer, not persisted to TB

            if len(optims) > 1:
                d_lr = optims[1].param_groups[0]["lr"]
                self.log("lr/d", d_lr, on_epoch=True, logger=False)
                _LOG.info(f"lr/g={g_lr:.8f} lr/d={d_lr:.8f} epoch={self.current_epoch}")
            else:
                _LOG.info(f"lr/g={g_lr:.8f} epoch={self.current_epoch}")
        else:
            g_lr = optims.param_groups[0]["lr"]
            self.log("lr/g", g_lr, on_epoch=True, logger=False)
            _LOG.info(f"lr/g={g_lr:.8f} epoch={self.current_epoch}")  

    def configure_optimizers(self):
        
        eps_cfg = self.hparams.adam_eps
        p = str(self.trainer.precision).lower()
        use_half = p.startswith("16") or p.startswith("bf16")   # matches "16", "16-mixed", "16-true", "bf16"
        eps = eps_cfg["fp16"] if use_half else eps_cfg["fp32"]

        # Instantiate optimiser & schedulers from the *factory*
        opt_g = self.optimizer_partial(params=self.autoencoder.parameters(), eps=eps)
        sch_g = self.scheduler_g_partial(optimizer=opt_g)

        if self.discriminator is not None and self.loss_manager_partial.keywords.get("adv_weight", 0.0) != 0:
            opt_d = self.optimizer_partial(params=self.discriminator.parameters(), eps=eps)
            sch_d = self.scheduler_d_partial(optimizer=opt_d)  
            return [opt_g, opt_d], [sch_g, sch_d]
        
        return [opt_g], [sch_g]
    
    def _tb_writer(self):
        logs = self.trainer.loggers if getattr(self.trainer, "loggers", None) else ([self.trainer.logger] if getattr(self.trainer, "logger", None) else [])
        for lg in logs:
            if isinstance(lg, pl.loggers.TensorBoardLogger):
                return lg.experiment
        return None


# Lightweight inference-only wrapper for autoencoders when running metrics.
# The adapter plugs arbitrary networks into the Lightning-based metrics pipeline
# without altering their original implementation.
class AutoencoderInferenceAdapter(StateDictLoaderMixin, pl.LightningModule):
    def __init__(
        self,
        autoencoder: torch.nn.Module,
        inferer: Optional[Any] = None,
        hparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            hparams,
            ignore=[
                "autoencoder",
                "inferer",
            ],
        )
        self.autoencoder = autoencoder
        self.inferer = inferer

    def _predict_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        """Return only a reconstruction tensor regardless of model output type."""
        out = self.autoencoder(x)

        if isinstance(out, dict) and "reconstruction" in out:
            recon_val = out["reconstruction"]
            if isinstance(recon_val, (list, tuple)):
                return recon_val[-1]
            return recon_val

        if isinstance(out, (list, tuple)):
            return out[0]

        return out

    def forward(self, images: torch.Tensor) -> Any:  # type: ignore[override]
        if self.inferer is not None:
            # Ensure the inferer sees a network that produces only reconstructions
            return self.inferer(inputs=images, network=self._predict_reconstruction)
            # return self.inferer(inputs=images, network=self.autoencoder)
            
        return self.autoencoder(images)

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        StateDictLoaderMixin.setup(self, stage)

    @torch.no_grad()
    def run_inference(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, None, None]:
        images = batch["image"]
        outputs = self.forward(images)

        if isinstance(outputs, dict) and "reconstruction" in outputs:
            recon_val = outputs["reconstruction"]
            if isinstance(recon_val, (list, tuple)):
                reconstruction = recon_val[-1]
            else:
                reconstruction = recon_val
        elif isinstance(outputs, (list, tuple)):
            reconstruction = outputs[0]
        else:
            reconstruction = outputs  # type: ignore[assignment]

        return reconstruction, None, None

