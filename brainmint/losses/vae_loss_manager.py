import torch
import logging
from pathlib import Path
from tabulate import tabulate
from typing import Dict, List

from .utils import KL_loss


class VaeLossManager:
    """
    Tracks, computes and averages losses for a VAE-GAN training loop.
    All running state lives on `device`.
    """
    
    _RECON_KEYS = ["recons", "kl", "perc"]
    _ADV_KEYS = ["adv_g", "adv_d"] 
    _STAT_KEYS = _RECON_KEYS + _ADV_KEYS

    def __init__(
        self,
        adv_loss_fn: torch.nn.Module,
        perc_loss_fn: torch.nn.Module,
        recon_loss_fn: torch.nn.Module,
        kl_weight: float = 0.0,
        adv_weight: float = 0.0,
        perceptual_weight: float = 0.0,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
    ) -> None:

        self.device = device

        # loss functions
        self.adv_loss_fn  = adv_loss_fn.to(self.device) if adv_loss_fn is not None else None
        self.perc_loss_fn = perc_loss_fn.eval().to(self.device) if perc_loss_fn is not None else None
        self.recon_loss_fn = recon_loss_fn.to(self.device) if recon_loss_fn is not None else torch.nn.L1Loss().to(self.device)

        # scalar weights
        self.kl_w = kl_weight
        self.adv_w = adv_weight
        self.perc_w = perceptual_weight
        
        # Epocs Loss record
        self.train_loss_rec: List[Dict[str, float]] = []
        self.val_loss_rec: List[Dict[str, float]] = []   

        self.logger = logger or logging.getLogger(__name__)
        self.logger.info( "LossManager: perc_w=%.4g kl_w=%.4g adv_w=%.4g", perceptual_weight, kl_weight, adv_weight)
        
        self.mode = "train"
        self.epoch_samples = {"train": 0, "val": 0}

        self.running_loss = { "train": {k: torch.zeros(1, device=self.device) for k in self._STAT_KEYS},
                                "val": {k: torch.zeros(1, device=self.device) for k in self._RECON_KEYS},}
        self.last_loss: Dict[str, Dict[str, torch.Tensor]]  = {}

        self._train_gen_calls = 0
        self._train_disc_calls = 0


    def reset_epoch(self,  mode: str | None = None):
        mode = mode or self.mode
        self.epoch_samples[mode] = 0
        if mode == "train":
            self._train_gen_calls = 0
            self._train_disc_calls = 0
        # running_loss entries are 1-element CUDA tensors | Accumulates Sample losses within Epoc
        keys = self._STAT_KEYS if mode == "train" else self._RECON_KEYS
        self.running_loss[mode] = {k: torch.zeros(1, device=self.device) for k in keys}
        self.last_loss[mode] = {}

    def set_mode(self, mode: str):
        assert mode in ("train", "val")
        self.mode = mode
    
    def get_train_loss_rec(self):
        return self.train_loss_rec

    def get_val_loss_rec(self):
        return self.val_loss_rec

    def weighted_sum(self, losses: dict) -> float:
        """
        Compute the weighted sum of reconstruction, KL, and perceptual losses.
        This is useful for val and checkpointing, so user code never needs to know the weights.
        """
        return (
            losses["recons"]
            + self.kl_w * losses["kl"]
            + self.perc_w * losses["perc"]
        )

    def weighted_sum_skip_kl(self, losses: dict) -> float:
        return losses["recons"] + self.perc_w * losses["perc"]

    def epoch_averages(self,  mode: str | None = None) -> Dict[str, float]:
        """Return CPU floats — safe for JSON/TensorBoard."""
        mode = mode or self.mode
        return {
            k: (v / max(1, self.epoch_samples[mode])).item()
            for k, v in self.running_loss[mode].items()
        }

    def check_consistency(self) -> None:
        """Raise if generator and discriminator were called different times."""
        if (self.adv_w > 0.0) and (self.adv_loss_fn is not None):
            if self._train_gen_calls != self._train_disc_calls:
                raise RuntimeError(
                    f"Generator called {self._train_gen_calls} times but discriminator "
                    f"{self._train_disc_calls} times in this epoch - something is off!"
                )


    def store_epoch(self, mode: str):
        """Archive current epoch averages into history and clear accumulated epoc loss and counters."""
        avg = self.epoch_averages(mode)
        if mode == "train":
            self.train_loss_rec.append(avg)
        elif mode == "val":
            self.val_loss_rec.append(avg)
        self.reset_epoch(mode)


    # Must be called at the end of batch step
    def inc_samples_count(self, bs: int):
        self.epoch_samples[self.mode] += bs

    # Accumulated Per Image Baises - As Multiplied by Batch Size. 
    @torch.no_grad()
    def _accumulate(self, comp: Dict[str, torch.Tensor], bs: int) -> None:
        buf = self.running_loss[self.mode]
        for k, v in comp.items():
            if k in buf:
                buf[k] += v.detach() * bs

    def compute_common_losses(
            self,
            images: torch.Tensor,
            recon: torch.Tensor,
            z_mu: torch.Tensor,
            z_sigma: torch.Tensor
        )-> Dict[str, torch.Tensor]:
        l_re  = self.recon_loss_fn(recon, images)
        l_kl  = KL_loss(z_mu, z_sigma) if (z_mu is not None and z_sigma is not None) else torch.tensor(0.0, device=self.device)
        l_pc  = self.perc_loss_fn(recon.float(), images.float()) if self.perc_loss_fn is not None else  torch.tensor(0.0, device=self.device)
        return {"recons": l_re, "kl": l_kl, "perc": l_pc}

    def gen_loss(
            self,
            images: torch.Tensor,
            recon: torch.Tensor,
            z_mu: torch.Tensor,
            z_sigma:torch.Tensor,
            logits_fake: torch.Tensor,
            bs: int,
            track_stats: bool = True,
        ) -> torch.Tensor:
        """Compute *total* generator loss (with weights) and optionally accumulate stats."""

        common_losses = self.compute_common_losses(images=images, recon=recon, z_mu=z_mu, z_sigma=z_sigma)
        
        total_loss_g  = ( 
            common_losses["recons"] +
            self.kl_w * common_losses["kl"] +
            self.perc_w * common_losses["perc"]
        )

        l_adv = torch.tensor(0.0, device=self.device)
        if self.adv_loss_fn is not None:       
            l_adv = self.adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)  
        total_loss_g += self.adv_w*l_adv
        
        self.last_loss[self.mode] = {**common_losses, "adv_g": l_adv}
        if track_stats:
            self._accumulate(self.last_loss[self.mode] , bs)
            self._train_gen_calls += 1
        
        return total_loss_g

    def disc_loss(
        self,
        logits_real: torch.Tensor,
        logits_fake: torch.Tensor,
        bs: int,
        track_stats: bool = True,
    ) -> torch.Tensor:
        """computes discriminator loss."""
        
        if self.adv_loss_fn is None:
            return torch.tensor(0.0, device=self.device)

        l_adv_d = 0.5 * (
            self.adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True) +
            self.adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
        )

        self.last_loss[self.mode]["adv_d"] = l_adv_d

        if track_stats:
            self._accumulate({"adv_d": l_adv_d}, bs)
            self._train_disc_calls += 1

        return l_adv_d

    def val_loss(
        self,
        images: torch.Tensor,
        recon: torch.Tensor,
        z_mu: torch.Tensor,
        z_sigma: torch.Tensor,
        bs: int,
        track_stats: bool = True,
    ) -> torch.Tensor:

        common_losses = self.compute_common_losses(images=images, recon=recon, z_mu=z_mu, z_sigma=z_sigma)
        
        total  = common_losses["recons"] + self.perc_w * common_losses["perc"]  # KL skipped

        self.last_loss[self.mode] = common_losses
        if track_stats:
            self._accumulate(self.last_loss[self.mode], bs)

        return total


def write_epoch_tables(
    loss_manager: VaeLossManager,
    table_path: Path,
    logger: logging.Logger | None = None,   
) -> None:
    """
    Save pretty Markdown tables of all recorded training/validation losses per epoch.
    """
    train_title = "Training Losses Per Epoch"
    val_title = "Validation Losses Per Epoch"
    
    train_rec = loss_manager.get_train_loss_rec()
    val_rec = loss_manager.get_val_loss_rec()
    logger = logger or logging.getLogger(__name__)

    with table_path.open("w") as f:
        
        # Training table
        if train_rec:
            headers = ["Epoch"] + list(train_rec[0].keys()) + ["Recon_total"]
            rows = []
            for i, average_losses in enumerate(train_rec):
                recon_total = loss_manager.weighted_sum(average_losses)
                row = [i] + [f"{average_losses[k]:.5f}" for k in average_losses.keys()] + [f"{recon_total:.5f}"]
                rows.append(row)
            f.write(f"## {train_title}\n")
            f.write(tabulate(rows, headers=headers, tablefmt="github"))
            f.write("\n\n")
        else:
            f.write(f"{train_title}: No records.\n\n")
        
        # Validation table
        if val_rec:
            headers = ["Epoch"] + list(val_rec[0].keys()) + ["Recon_total"]
            rows = []
            for i, average_losses in enumerate(val_rec):
                recon_total = loss_manager.weighted_sum_skip_kl(average_losses)
                row = [i] + [f"{average_losses[k]:.5f}" for k in average_losses.keys()] + [f"{recon_total:.5f}"]
                rows.append(row)
            f.write(f"## {val_title}\n")
            f.write(tabulate(rows, headers=headers, tablefmt="github"))
            f.write("\n")
        else:
            f.write(f"{val_title}: No records.\n")

    logger.info(f"Wrote epoch tables to {table_path}")
