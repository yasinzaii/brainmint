# ruff: noqa: E402
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "configs"

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from monai.data import DataLoader
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from omegaconf import OmegaConf, open_dict

from brainmint.lightning.diffusion_module import DiffusionModule


def _make_unet(num_classes: int = 4):
    """Instantiate a lightweight MAISI diffusion UNet."""

    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="model/diffusion/maisi",
            overrides=[
                "model.diffusion.diffusion_maisi.use_flash_attention=false",
                "model.shared.latent_channels=1",
                "model.diffusion.diffusion_maisi.num_channels=[8,16,16,32]",
                "model.diffusion.diffusion_maisi.norm_num_groups=4",
                "model.diffusion.diffusion_maisi.num_head_channels=[0,0,4,4]",
                f"model.diffusion.diffusion_maisi.num_class_embeds={num_classes}",
            ],
        )
    return instantiate(cfg.model.diffusion.diffusion_maisi)



class _RandomLatentWithModalityDataset:
    """Returns dict with 'latent', 'image', 'modality' and modality_map labels."""

    def __init__(self, modalities, shape=(1, 8, 8, 8)):
        self.modalities = list(modalities)
        self.shape = shape

    def __len__(self):
        return len(self.modalities)

    def __getitem__(self, idx):
        vol = torch.randn(self.shape, dtype=torch.float32)
        return {
            "latent": vol,
            "image": vol,
            "modality": self.modalities[idx],
            "modality_map": torch.tensor(idx, dtype=torch.long),
        }


class Tiny3DDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.modalities = ["T1w", "T2w"]
        self.setup()

    def setup(self, stage=None):
        ds = _RandomLatentWithModalityDataset(self.modalities, shape=(1, 8, 8, 8))
        self.train_ds = ds
        self.val_ds = ds
        self.test_ds = ds

    def _dl(self, ds):
        return DataLoader(ds, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def train_dataloader(self): return self._dl(self.train_ds)
    def val_dataloader(self):   return self._dl(self.val_ds)
    def test_dataloader(self):  return self._dl(self.test_ds)


latent_specs = '[{key:latent,group:latent,modalities:[t1w,t2w]}]'


def test_prep_condition_requires_modality_map():
    module = DiffusionModule(
        autoencoder=None,
        diffusion_unet=_make_unet(num_classes=4),
        noise_scheduler=nn.Identity(),
        hparams={
            "lr": 1e-3,
            "loss": "L1",
            "latent_source": "mu",
            "scale_factor_batches": 1,
            "scale_factor": 1.0,
            "num_inference_steps": 2,
            "input_key": "latent",
            "latent_channels": 1,
        },
        optimizer=lambda params: torch.optim.SGD(params, lr=1e-3),
    )
    good = {
        "latent": torch.randn(2, 1, 4, 4, 4),
        "modality_map": torch.tensor([3, 2], dtype=torch.long),
    }
    cond = module._prep_condition_tensors(good)
    assert torch.equal(cond["class_labels"], good["modality_map"])

    bad = {"latent": torch.randn(2, 1, 4, 4, 4)}
    with pytest.raises(ValueError):
        module._prep_condition_tensors(bad)


def test_sample_latent_passes_class_labels():
    unet = _make_unet(num_classes=4)

    class _RecordWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.last_labels = None

        def forward(self, x, timesteps, class_labels=None, **kwargs):
            self.last_labels = class_labels.detach().cpu() if class_labels is not None else None
            return self.base(x, timesteps, class_labels=class_labels, **kwargs)

        @property
        def num_class_embeds(self):
            return getattr(self.base, "num_class_embeds", None)

        @property
        def in_channels(self):
            return getattr(self.base, "in_channels", 1)

    wrapped = _RecordWrapper(unet)
    module = DiffusionModule(
        autoencoder=None,
        diffusion_unet=wrapped,
        noise_scheduler=RFlowScheduler(
            num_train_timesteps = 1000,
            use_discrete_timesteps = False,
            use_timestep_transform = True,
            sample_method = "uniform",
            scale = 1.4,
        ),
        hparams={
            "lr": 1e-3,
            "loss": "L1",
            "latent_source": "mu",
            "scale_factor_batches": 1,
            "scale_factor": 1.0,
            "num_inference_steps": 2,
            "input_key": "latent",
            "latent_channels": 1,
        },
        optimizer=lambda params: torch.optim.SGD(params, lr=1e-3),
    )
    module.eval()
    labels = torch.tensor([0, 1], dtype=torch.long)
    z = module.sample_latent((8, 8, 8), batch_size=2, class_labels=labels)
    assert torch.equal(wrapped.last_labels, labels)
    assert z.shape == (2, 1, 8, 8, 8)

    with pytest.raises(ValueError):
        module.sample_latent((8, 8, 8), batch_size=2, class_labels=torch.tensor([0], dtype=torch.long))


def test_prediction_type_override():
    class _Sched(nn.Module):
        def __init__(self):
            super().__init__()
            self.prediction_type = DDPMPredictionType.EPSILON

        def add_noise(self, original_samples, noise, timesteps):
            return original_samples

        def sample_timesteps(self, z):
            return torch.zeros(z.shape[0], dtype=torch.long)

    module = DiffusionModule(
        autoencoder=None,
        diffusion_unet=_make_unet(num_classes=4),
        noise_scheduler=_Sched(),
        hparams={
            "lr": 1e-3,
            "loss": "L1",
            "latent_source": "mu",
            "scale_factor_batches": 1,
            "scale_factor": 1.0,
            "num_inference_steps": 1,
            "input_key": "latent",
            "prediction_type": "sample",
            "latent_channels": 1,
        },
        optimizer=lambda params: torch.optim.SGD(params, lr=1e-3),
    )
    assert module.prediction_type == DDPMPredictionType.SAMPLE
    assert module.noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE


def test_encode_using_autoencoder_flag():
    class _AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def encode(self, x):
            self.called = True
            return x, None

    sched = nn.Module()
    sched.num_train_timesteps = 1

    def _add_noise(original_samples, noise, timesteps):
        return original_samples

    sched.add_noise = _add_noise

    batch = {"image": torch.randn(1, 1, 8, 8, 8), "modality_map": torch.zeros(1, dtype=torch.long)}

    ae1 = _AE()
    module1 = DiffusionModule(
        autoencoder=ae1,
        diffusion_unet=_make_unet(num_classes=4),
        noise_scheduler=sched,
        hparams={
            "lr": 1e-3,
            "loss": "L1",
            "latent_source": "mu",
            "scale_factor_batches": 1,
            "scale_factor": 1.0,
            "num_inference_steps": 1,
            "input_key": "image",
            "encode_using_autoencoder": False,
            "latent_channels": 1,
        },
        optimizer=lambda params: torch.optim.SGD(params, lr=1e-3),
    )
    module1.training_step(batch, 0)
    assert not ae1.called

    ae2 = _AE()
    module2 = DiffusionModule(
        autoencoder=ae2,
        diffusion_unet=_make_unet(num_classes=4),
        noise_scheduler=sched,
        hparams={
            "lr": 1e-3,
            "loss": "L1",
            "latent_source": "mu",
            "scale_factor_batches": 1,
            "scale_factor": 1.0,
            "num_inference_steps": 1,
            "input_key": "image",
            "encode_using_autoencoder": True,
            "latent_channels": 1,
        },
        optimizer=lambda params: torch.optim.SGD(params, lr=1e-3),
    )
    module2.training_step(batch, 0)
    assert ae2.called


def test_run_inference_accepts_str_labels():
    unet = _make_unet(num_classes=4)
    module = DiffusionModule(
        autoencoder=None,
        diffusion_unet=unet,
        noise_scheduler=RFlowScheduler(
            num_train_timesteps = 1000,
            use_discrete_timesteps = False,
            use_timestep_transform = True,
            sample_method = "uniform",
            scale = 1.4,
        ),
        hparams={
            "lr": 1e-3,
            "loss": "L1",
            "latent_source": "mu",
            "scale_factor_batches": 1,
            "scale_factor": 1.0,
            "num_inference_steps": 2,
            "input_key": "latent",
            "modality_map": {"T1w": 0, "T2w": 1},
            "latent_channels": 1,
        },
        optimizer=lambda params: torch.optim.SGD(params, lr=1e-3),
    )
    module.eval()
    x = torch.randn(1, 1, 8, 8, 8)
    res = module.run_inference(x, class_labels="T2w")
    assert set(res.keys()) == {"sample", "latent"}


def test_run_inference_rejects_invalid_labels():
    module = DiffusionModule(
        autoencoder=None,
        diffusion_unet=_make_unet(num_classes=4),
        noise_scheduler=RFlowScheduler(
            num_train_timesteps = 1000,
            use_discrete_timesteps = False,
            use_timestep_transform = True,
            sample_method = "uniform",
            scale = 1.4,
        ),
        hparams={
            "lr": 1e-3,
            "loss": "L1",
            "latent_source": "mu",
            "scale_factor_batches": 1,
            "scale_factor": 1.0,
            "num_inference_steps": 2,
            "input_key": "latent",
            "modality_map": {"T1w": 0, "T2w": 1},
            "latent_channels": 1,
        },
        optimizer=lambda params: torch.optim.SGD(params, lr=1e-3),
    )
    module.eval()
    x = torch.randn(1, 1, 8, 8, 8)
    with pytest.raises(ValueError):
        module.run_inference(x, class_labels="T3w")
    with pytest.raises(ValueError):
        module.run_inference(x, class_labels=torch.tensor([5], dtype=torch.long))

@pytest.mark.parametrize("input_specs", [latent_specs])
def test_diffusion_fast_loop(tmp_path, input_specs):
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        img_keys = "[image]" if "image" in input_specs else "[]"
        label_keys = "[label]" if "label" in input_specs else "[]"
        latent_keys = "[latent]" if "latent" in input_specs else "[]"

        cfg = compose(
            config_name="exp/maisi/train_diffusion",
            overrides=[
                "dataset.brainscape.dataset_root=${paths.masi_brainscape_prep_test}",
                "dataset.brainscape.json_path=${paths.masi_brainscape_json_test}",
                "dataset.brainscape.num_workers=0",
                f"dataset.brainscape.input_specs={input_specs}",
                "dataset.brainscape.subset_frac=null",
                f"dataset.train_tf.image_keys={img_keys}",
                f"dataset.train_tf.label_keys={label_keys}",
                f"dataset.train_tf.passthrough_keys={latent_keys}",
                "dataset.train_tf.meta_keys=[modality]",
                "dataset.conditioning.modality_conditioning.key_name=modality_map",
    
                # PL Trainer Overrides
                "trainer.pl_trainer.devices=1",
                "trainer.pl_trainer.precision=32-true",
                "trainer.pl_trainer.max_epochs=3",
                "trainer.pl_trainer.log_every_n_steps=1",
                "trainer.pl_trainer.check_val_every_n_epoch=1",
                "trainer.pl_trainer.log_every_n_steps=1", 

                # Lightning Module Settings
                "lightning.diffusion_lightning.autoencoder=null",
                "lightning.diffusion_lightning.hparams.input_key=latent",
                "lightning.diffusion_lightning.hparams.num_inference_steps=1",
                "lightning.diffusion_lightning.hparams.weight_loads=[]",
                "lightning.diffusion_lightning.hparams.modality_map={t1w:0,t2w:1}",
                "lightning.diffusion_lightning.hparams.latent_channels=1",
                "fast_dev_run=false",
                f"hydra.run.dir={tmp_path.as_posix()}",
                "hydra.job.name=test_diffusion",
                "model.diffusion.diffusion_maisi.use_flash_attention=false",
                "model.shared.latent_channels=1",
            ],
        )

        with open_dict(cfg):
            if "callbacks" in cfg and cfg.callbacks is not None:
                for cb in cfg.callbacks.get("diff_callbacks", []):
                    tgt = cb.get("_target_", "")
                    if "dirpath" in cb:
                        if tgt.endswith("ModelCheckpoint"):
                            cb.dirpath = f"{tmp_path.as_posix()}/checkpoints"
                        else:
                            cb.dirpath = tmp_path.as_posix()
                    if cb.get("monitor") == "val_total":
                        cb.monitor = "val/loss"
            if "loggers" in cfg and cfg.loggers is not None:
                for lg in cfg.loggers:
                    if "save_dir" in lg:
                        lg.save_dir = tmp_path.as_posix()

        module = instantiate(cfg.lightning.diffusion_lightning)
        dm = Tiny3DDataModule(batch_size=2)

        print(OmegaConf.to_yaml(cfg, resolve=True))

        callbacks = [instantiate(c) for c in cfg.callbacks.diff_callbacks]
        loggers   = [instantiate(logger_cfg) for logger_cfg in cfg.loggers]

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

