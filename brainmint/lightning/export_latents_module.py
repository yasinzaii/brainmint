import shutil
import logging

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from monai.data import MetaTensor
from monai.transforms import SaveImage

from brainmint.inference.dynamic_inference import dynamic_infer
from brainmint.utils.state_dict_loader import StateDictLoaderMixin

_LOG = logging.getLogger(__name__)

class ExportLatentsModule(StateDictLoaderMixin, pl.LightningModule):
    """
    Predict-only: for each preprocessed input
      • copy input → out_dataset_dir/<DATASET>/preprocessed/<rel>.nii.gz
      • write z_mu  → *_emb.nii.gz
      • write recon → *_recon.nii.gz
    Also accumulates global stats of z_mu for a dataset-level scale_factor.
    """

    def __init__(
        self,
        autoencoder,             # Instantiated VAE autoencoder
        inferer,                 # Instantiated (e.g., SlidingWindowInferer)
        src_dataset_root: str,   # Original BrainScape prep <root>
        out_dataset_dir: str,    # Output dataset <root>
        save_sigma: bool = False,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        enable_clamping: bool = False,
        hparams: Dict[str, Any] = None,  # weight_loads for StateDictLoaderMixin
    ):
        super().__init__()
        self.save_hyperparameters(hparams, ignore=["autoencoder", "inferer"])

        self.autoencoder = autoencoder
        self.inferer = inferer
        self.src_dataset_root = Path(src_dataset_root).resolve()
        self.out_dataset_dir  = Path(out_dataset_dir).resolve()
        if self.src_dataset_root == self.out_dataset_dir:
            raise ValueError("out_dataset_dir must differ from src_dataset_root to avoid overwriting source data")
        self.save_sigma = bool(save_sigma)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.enable_clamping = bool(enable_clamping)

        # Global accumulators
        self.register_buffer("_sum_sq", torch.zeros([], dtype=torch.float64))
        self.register_buffer("_count",  torch.zeros([], dtype=torch.float64))

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad = False
    
    def _dst_abs_for_src(self, src_abs: Path) -> tuple[Path, Path]:
        """
        Map a source file under src_dataset_root to its destination absolute path in out_dataset_dir.
        Returns (dst_abs, dst_dir). Enforces expected layout: <DATASET>/preprocessed/<rel>
        """
        try:
            rel = src_abs.resolve().relative_to(self.src_dataset_root)
        except Exception as e:
            raise RuntimeError(f"Input {src_abs} is not under src_dataset_root={self.src_dataset_root}") from e

        parts = rel.parts
        if len(parts) < 3 or parts[1] != "preprocessed":
            raise RuntimeError(f"Unexpected path layout: {rel} (expected <DATASET>/preprocessed/<rel_path>)")

        dst_abs = (self.out_dataset_dir / rel).resolve()
        return dst_abs, dst_abs.parent
            
    # Autoencoder 16-Mixed ? -> 32-True
    def on_predict_start(self):
        # AE CKPT already loaded - ensure FP32 weights and disable any FP16 flags
        self.autoencoder.to(dtype=torch.float32)
        print("Autoencoder - turning off FP16 settings", flush=True)
        for m in self.autoencoder.modules():
            if hasattr(m, "norm_float16"):
                m.norm_float16 = False


    @torch.no_grad()
    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        imgs = batch["image"].to(self.device, non_blocking=True)  # MetaTensor list or batch
        recon, z_mu, z_sigma = dynamic_infer(inferer=self.inferer, model=self.autoencoder, images=imgs)
        if self.enable_clamping:
            recon = recon.clamp(self.clamp_min, self.clamp_max)

        # Accumulate stats
        z_mu_f = z_mu.detach().to(torch.float32)
        batch_sum_sq = (z_mu_f.double() ** 2).sum()
        batch_count  = torch.tensor(z_mu_f.numel(), dtype=torch.float64, device=self._sum_sq.device)
        self._sum_sq += batch_sum_sq
        self._count  += batch_count

        # Save per-sample files alongside the source preprocessed image
        bs = imgs.shape[0]
        for i in range(bs):
            src_meta = getattr(imgs[i], "meta", {}) or {}
            src_abs  = Path(str(src_meta.get("filename_or_obj"))).resolve()
            dst_abs, dst_dir = self._dst_abs_for_src(src_abs)

            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_abs, dst_abs)

            def _save_next_to_dst(
                t: torch.Tensor,
                postfix: str,
                meta: Optional[Dict[str, Any]] = None,
            ) -> None:
                # Use source name as base, but write in dst_dir; SaveImage uses base from meta filename
                base_meta = dict(meta) if meta else {}
                base_meta["filename_or_obj"] = str(dst_abs)
                mt = MetaTensor(t.detach().cpu(), meta=base_meta)
                SaveImage(
                    output_dir=str(dst_dir),
                    output_postfix=postfix,   # "emb_mu" or "recon"
                    output_ext=".nii.gz",
                    separate_folder=False,
                )(mt)

            _save_next_to_dst(z_mu[i], postfix="emb_mu")
            _save_next_to_dst(recon[i], postfix="recon", meta=src_meta)

            if self.save_sigma and z_sigma is not None:
                _save_next_to_dst(z_sigma[i], postfix="emb_sigma")

        return None
