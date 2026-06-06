import sys
from pathlib import Path

import torch
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict, OmegaConf
import pytorch_lightning as pl
from monai.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "configs"


def _make_module(tmp_path):
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="exp/maisi/train_controlnet",
            overrides=[
                "model.shared.latent_channels=1",
                
                "model.diffusion.diffusion_maisi.use_flash_attention=false",
                "model.diffusion.diffusion_maisi.num_channels=[8,16]",
                "model.diffusion.diffusion_maisi.attention_levels=[false,false]",
                "model.diffusion.diffusion_maisi.num_head_channels=[4,4]",
                "model.diffusion.diffusion_maisi.norm_num_groups=4",
                "model.controlnet.controlnet_maisi.num_class_embeds=4",

                "model.controlnet.controlnet_maisi.num_channels=[8,16]",
                "model.controlnet.controlnet_maisi.attention_levels=[false, false]",
                "model.controlnet.controlnet_maisi.num_head_channels=[4,4]",
                "model.controlnet.controlnet_maisi.norm_num_groups=4",
                "model.controlnet.controlnet_maisi.conditioning_embedding_in_channels=5",
                "model.controlnet.controlnet_maisi.conditioning_embedding_num_channels=[4, 8, 64]",
                "model.controlnet.controlnet_maisi.num_class_embeds=4",
                
                "lightning.controlnet_lightning.hparams.latent_channels=1",
                "lightning.controlnet_lightning.hparams.modality_map={t1w:0,t2w:1}",
                "lightning.controlnet_lightning.autoencoder=null",
                "lightning.controlnet_lightning.hparams.scale_factor=1.0",

                f"hydra.run.dir={tmp_path.as_posix()}",
                "hydra.job.name=test_controlnet",
            ],
        )
        
        module = instantiate(cfg.lightning.controlnet_lightning)
        module.noise_scheduler.sample_random_times = (
            lambda batch_size, device: torch.randint(
                0, module.noise_scheduler.num_train_timesteps, (batch_size,), device=device
            )
        )
    return module


def _make_batch(batch_size: int = 2):
    return {
        "latent": torch.randn(batch_size, 1, 4, 4, 4),
        "mask_one_hot": torch.zeros(batch_size, 4, 16, 16, 16),
        "image_masked": torch.zeros(batch_size, 1, 16, 16, 16),
        "modality_map": torch.arange(batch_size, dtype=torch.long),
    }


def test_controlnet_training_step_runs(tmp_path):
    module = _make_module(tmp_path)
    batch = _make_batch()
    loss = module.training_step(batch, 0)
    assert torch.is_tensor(loss) and loss.shape == ()


def test_controlnet_requires_masks(tmp_path):
    module = _make_module(tmp_path)
    batch = _make_batch()
    batch.pop("mask_one_hot")
    with pytest.raises(KeyError):
        module.training_step(batch, 0)


def test_controlnet_run_inference_label_validation(tmp_path):
    module = _make_module(tmp_path)
    x = torch.randn(1, 1, 4, 4, 4)
    with pytest.raises(ValueError):
        module.run_inference(x, class_labels="T3w")
    with pytest.raises(ValueError):
        module.run_inference(x, class_labels=torch.tensor([5], dtype=torch.long))


class _RandomControlNetDataset:
    """Synthetic dataset yielding the keys required by ControlNetModule."""

    def __init__(self, modalities, shape=(1, 8, 8, 8)):
        self.modalities = list(modalities)
        self.shape = shape

    def __len__(self):
        return len(self.modalities)

    def __getitem__(self, idx):
        vol = torch.randn(self.shape, dtype=torch.float32)
        spatial = self.shape[-3:]
        latent = torch.randn((4, *(tuple(int(x / 4) for x in spatial))), dtype=torch.float32)
        mask = torch.zeros((4, *spatial), dtype=torch.float32)
        mask[0] = 1.0
        return {
            "latent": latent.clone(),
            "mask_one_hot": mask,
            "image_masked": vol.clone(),
            "modality": self.modalities[idx],
            "modality_map": torch.tensor(idx, dtype=torch.long),
        }


class TinyControlNetDataModule(pl.LightningDataModule):
    """Minimal datamodule serving synthetic volumes for testing."""

    def __init__(self, batch_size: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.modalities = ["T1w", "T2w"]
        self.setup()

    def setup(self, stage=None):
        ds = _RandomControlNetDataset(self.modalities, shape=(1, 32, 32, 32))
        self.train_ds = ds
        self.val_ds = ds

    def _dl(self, ds):
        return DataLoader(ds, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def train_dataloader(self):
        return self._dl(self.train_ds)

    def val_dataloader(self):
        return self._dl(self.val_ds)


@pytest.mark.parametrize("latent_channels", [4])
def test_controlnet_fast_loop(tmp_path, latent_channels):
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="exp/maisi/train_controlnet",
            overrides=[
                "model.shared.latent_channels={0}".format(latent_channels),
                
                "model.autoencoder.autoencoder_maisi.norm_float16=false",

                "model.diffusion.diffusion_maisi.use_flash_attention=false",
                # "model.diffusion.diffusion_maisi.num_channels=[8,16]",
                # "model.diffusion.diffusion_maisi.attention_levels=[false,false]",
                # "model.diffusion.diffusion_maisi.num_head_channels=[4,4]",
                # "model.diffusion.diffusion_maisi.norm_num_groups=4",
                
                # "model.controlnet.controlnet_maisi.num_channels=[8,16]",
                # "model.controlnet.controlnet_maisi.attention_levels=[false,false]",
                # "model.controlnet.controlnet_maisi.num_head_channels=[4,4]",
                # "model.controlnet.controlnet_maisi.norm_num_groups=4",
                # "model.controlnet.controlnet_maisi.conditioning_embedding_in_channels=5",
                "model.controlnet.controlnet_maisi.conditioning_embedding_num_channels=[4, 4, 64]",
                # "model.controlnet.controlnet_maisi.num_class_embeds=4",
                
                "lightning.controlnet_lightning.hparams.latent_channels={0}".format(latent_channels),
                "lightning.controlnet_lightning.hparams.scale_factor=1.0",
                "lightning.controlnet_lightning.hparams.modality_map={t1w:0,t2w:1}",
 
                "trainer.pl_trainer.devices=1",
                "trainer.pl_trainer.precision=32-true",
                "trainer.pl_trainer.max_epochs=2",
                "trainer.pl_trainer.log_every_n_steps=1",
                "trainer.pl_trainer.check_val_every_n_epoch=1",
                "fast_dev_run=false",
                f"hydra.run.dir={tmp_path.as_posix()}",
                "hydra.job.name=test_controlnet",
            ],
        )

        with open_dict(cfg):
            if "callbacks" in cfg and cfg.callbacks is not None:
                cbs = cfg.callbacks.get("controlnet_callbacks", [])
                for cb in cbs:
                    tgt = cb.get("_target_", "")
                    if "dirpath" in cb:
                        if tgt.endswith("ModelCheckpoint"):
                            cb.dirpath = f"{tmp_path.as_posix()}/checkpoints"
                        else:
                            cb.dirpath = tmp_path.as_posix()
                    if cb.get("monitor") == "val_total":
                        cb.monitor = "val/loss"
            if cfg.get("loggers"):
                for lg in cfg.loggers:
                    if "save_dir" in lg:
                        lg.save_dir = tmp_path.as_posix()

        module = instantiate(cfg.lightning.controlnet_lightning)
        dm = TinyControlNetDataModule(batch_size=2)

        print(OmegaConf.to_yaml(cfg, resolve=True))

        callbacks = []
        if cfg.get("callbacks"):
            for cb in cfg.callbacks.get("controlnet_callbacks", []):
                callbacks.append(instantiate(cb))
        loggers = []
        if cfg.get("loggers"):
            for lg in cfg.loggers:
                loggers.append(instantiate(lg))

        trainer = pl.Trainer(
            **cfg.trainer.pl_trainer,
            default_root_dir=str(tmp_path),
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            logger=loggers,
        )
        trainer.fit(module, datamodule=dm)

    run_dir = tmp_path
    ckpt_dirs = list((run_dir / "checkpoints").rglob("*"))
    assert ckpt_dirs, "no checkpoints directory"
    tb_dir = run_dir / "tensorboard"
    assert tb_dir.exists(), "no tensorboard logs"
    samples_dir = run_dir / "val_samples" / "epoch-000"
    assert samples_dir.exists(), f"{samples_dir} missing (SaveMRIImages not triggered?)"
    any_sample = any(p.name.endswith("_sample.nii.gz") for p in samples_dir.glob("*_sample.nii.gz"))
    any_latent = any(p.name.endswith("_latent.nii.gz") for p in samples_dir.glob("*_latent.nii.gz"))
    assert any_sample and any_latent, "expected sample and latent output niftis"
