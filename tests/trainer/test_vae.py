import sys
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from monai.data import CacheDataset, DataLoader
from omegaconf import OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "configs"


# Tiny synthetic DataModule (2 samples, 1 batch)
class _RandomVolumeDataset(CacheDataset):
    """Returns dicts that mimic BrainScape dataloader output."""
    def __init__(self, n=2, shape=(1, 16, 16, 16)):
        self.n, self.shape = n, shape

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"image": torch.randn(self.shape, dtype=torch.float32)}

class _RandomVolumeWithModalityDataset:
    """Returns dict with 'image' and 'modality' keys to satisfy SaveMRIImages."""
    def __init__(self, modalities, shape=(1, 16, 16, 16)):
        self.modalities = list(modalities)
        self.shape = shape

    def __len__(self):
        return len(self.modalities)

    def __getitem__(self, idx):
        return {"image": torch.randn(self.shape, dtype=torch.float32), "modality": self.modalities[idx]}

class Tiny3DDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 2, mri_shape = (1, 16, 16, 16)):
        super().__init__()
        self.batch_size = batch_size
        self.modalities = ["T1w", "T2w"]
        self.mri_shape = mri_shape
        self.mri_shape = mri_shape
        self.setup()
        

    def setup(self, stage = None):
        self.train_ds = _RandomVolumeDataset(n=2, shape=self.mri_shape )
        self.val_ds   = _RandomVolumeWithModalityDataset(self.modalities, self.mri_shape )
        self.test_ds  = _RandomVolumeWithModalityDataset(self.modalities, self.mri_shape )

    def _dl(self, ds):
        return DataLoader(ds, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def train_dataloader(self): return self._dl(self.train_ds)
    def val_dataloader(self):   return self._dl(self.val_ds)
    def test_dataloader(self):  return self._dl(self.test_ds)


@pytest.mark.parametrize("shape", [(1, 32, 32, 32)])
def test_vae_fast_loop(tmp_path, shape):
    """
    Build VAEModule from Hydra config, run one quick CPU epoch with:
      • ModelCheckpoint (best-k + every-epoch),
      • LearningRateMonitor, ModelSummary,
      • SaveMRIImages (writes NIfTIs per modality),
      • TensorBoard + CSV loggers.
    Assert: forward sanity, metrics logged, checkpoints saved, images written.
    """

    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="exp/maisi/train_vae",
            overrides=[
                "trainer.pl_trainer.devices=1",
                "trainer.pl_trainer.precision=32-true",
                "model.autoencoder.autoencoder_maisi.norm_float16=false",
                "trainer.pl_trainer.max_epochs=1",
                "trainer.pl_trainer.log_every_n_steps=1",
                "trainer.pl_trainer.check_val_every_n_epoch=1",
                "fast_dev_run=false",
                
                # Put the whole Hydra run in tmp_path 
                f"hydra.run.dir={tmp_path.as_posix()}",
                "hydra.job.name=test_vae",

                # Minimal loss functions
                "loss.vae_loss.perc_loss_fn=null",
                "loss.vae_loss.perceptual_weight=0.0",
            ],
        )

        with open_dict(cfg):
            if "callbacks" in cfg and cfg.callbacks is not None:
                for cb in cfg.callbacks.get("vae_callbacks", []):
                    tgt = cb.get("_target_", "")
                    if tgt.endswith("SaveMRIImages"):
                        cb.dirpath = tmp_path.as_posix()
                        cb.output_names = ["_ouput_test", "mu", "sigma"]
                    elif "dirpath" in cb:  # ModelCheckpoint entries have 'dirpath'
                        cb.dirpath = f"{tmp_path.as_posix()}/checkpoints"

            if "loggers" in cfg and cfg.loggers is not None:
                for lg in cfg.loggers:
                    if "save_dir" in lg:
                        lg.save_dir = tmp_path.as_posix()

        module = instantiate(cfg.lightning.vae_lightning)
        dm = Tiny3DDataModule(batch_size=2, mri_shape=shape)
        dm = Tiny3DDataModule(batch_size=2, mri_shape=shape)

        print(OmegaConf.to_yaml(cfg, resolve=True))

        # Instantiate callbacks/loggers from Hydra config
        callbacks = [instantiate(c) for c in cfg.callbacks.vae_callbacks]
        loggers   = [instantiate(logger_cfg) for logger_cfg in cfg.get("loggers", [])]

        batch = next(iter(dm.train_dataloader()))
        x = batch["image"]
        recon, z_mu, z_sigma = module(x)
        assert recon.shape == x.shape and torch.isfinite(recon).all()
        assert module.automatic_optimization is False

        trainer = pl.Trainer(
            **cfg.trainer.pl_trainer,
            default_root_dir=str(tmp_path),
            limit_train_batches=2,
            limit_val_batches=2,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            logger=loggers,
        )
        trainer.fit(module, datamodule=dm)

    run_dir = tmp_path
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    csv_dir = run_dir / "csv_logs" / "version_0"
    val_samples = run_dir / "val_samples" / "epoch-000"

    assert ckpt_dir.exists(), "checkpoints dir missing"
    epoch_ckpts = list(ckpt_dir.glob("epoch*.ckpt"))
    assert any(p.name.startswith("epoch000") for p in epoch_ckpts), "epoch000.ckpt not found (every-epoch checkpoint)"
    assert (ckpt_dir / "last.ckpt").exists(), "last.ckpt not found"

    assert tb_dir.exists() and any(tb_dir.rglob("events.*")), "TensorBoard events not written"

    metrics_csv = csv_dir / "metrics.csv"
    assert metrics_csv.exists() and metrics_csv.stat().st_size > 0, "CSV metrics not written"

    assert val_samples.exists(), f"{val_samples} missing (SaveMRIImages not triggered?)"
    any_input = any(p.name.endswith("_input.nii.gz") for p in val_samples.glob("*.nii.gz"))
    any_output = any("_ouput_test.nii.gz" in p.name for p in val_samples.glob("*.nii.gz"))
    assert any_input and any_output, "expected input/output nifti files in val_samples/epoch-000"
