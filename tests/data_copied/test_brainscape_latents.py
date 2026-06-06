import sys
import json
import nibabel as nib
from pathlib import Path

import torch
import pytest
import numpy as np
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "configs"

from brainmint.lightning.export_latents_module import ExportLatentsModule
from experiments.maisi.export_latents import _clone_json_with_latents

def _write_nifti(path: Path, shape=(32, 32, 32)) -> None:
    img = nib.Nifti1Image(np.random.rand(*shape).astype("float32"), affine=np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)

def _generate_dummy_latent_dataset(tmp_path: Path) -> tuple[Path, Path]:
    src_root = tmp_path / "src"
    out_root = tmp_path / "out"
    src_json = tmp_path / "src.json"
    out_json = tmp_path / "out.json"

    data = {
        "train": [
            {"dataset": "BRATS", "subject": "s1", "preprocessed": {"t1w": "s1_t1w.nii.gz", "t2w": "s1_t2w.nii.gz"}},
            {"dataset": "AOMIC", "subject": "s2", "preprocessed": {"t1w": "s2_t1w.nii.gz", "flair": "s2_flair.nii.gz"}},
        ],
        "val": [
            {"dataset": "HCP", "subject": "v1", "preprocessed": {"t2w": "v1_t2w.nii.gz", "t1ce": "v1_t1ce.nii.gz"}},
        ],
        "test": [
            {"dataset": "FC100", "subject": "t1", "preprocessed": {"t1w": "t1_t1w.nii.gz"}},
        ],
    }

    with src_json.open("w") as f:
        json.dump(data, f)

    for split in data.values():
        for rec in split:
            for rel in rec["preprocessed"].values():
                _write_nifti(src_root / rec["dataset"] / "preprocessed" / rel)

    
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            # Dataset Overrides
            f"paths.brainscape_prep={src_root}",
            f"paths.brainscape_json={src_json}",
            f"paths.masi_brainscape_prep={out_root}",
            f"paths.masi_brainscape_json={out_json}",
            "dataset.brainscape.num_workers=0",
            "dataset.brainscape.subset_frac=null",
            "dataset.brain_roi_size=[32,32,32]",

            # Trainer Overrides
            "trainer.pl_trainer.accelerator=cpu",
            "trainer.pl_trainer.devices=1",

            # AE Model Overrides
            "model.autoencoder.autoencoder_maisi.num_channels=[4,8]",
            "model.autoencoder.autoencoder_maisi.num_res_blocks=[1,1]",
            "model.autoencoder.autoencoder_maisi.latent_channels=2",
            "model.autoencoder.autoencoder_maisi.norm_float16=false",
            "model.autoencoder.autoencoder_maisi.norm_num_groups=1",
            "model.autoencoder.autoencoder_maisi.attention_levels=[false,false]",
        ]

        cfg = compose(config_name="exp/maisi/brainscape_latents", overrides=overrides)
        dm = instantiate(cfg.dataset.brainscape)
        dm.setup()

        inferer = instantiate(cfg.lightning.export_latents_module.inferer)
        autoencoder = instantiate(cfg.model.autoencoder.autoencoder_maisi)
        module = ExportLatentsModule(
            autoencoder=autoencoder,
            inferer=inferer,
            src_dataset_root=str(src_root),
            out_dataset_dir=str(out_root),
            save_sigma=False,
            clamp_min=0.0,
            clamp_max=1.0,
        )
        trainer = pl.Trainer(**cfg.trainer.pl_trainer, logger=False, enable_checkpointing=False)
        trainer.predict(module, dataloaders=dm.train_dataloader())
        trainer.predict(module, dataloaders=dm.val_dataloader())
        trainer.predict(module, dataloaders=dm.test_dataloader())

    _clone_json_with_latents(src_json=src_json, dst_json=out_json, dataset_root=out_root)
    return out_root, out_json


def test_brainscape_latent_only(tmp_path):
    root, json_path = _generate_dummy_latent_dataset(tmp_path)
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            # Dataset Overrides
            f"paths.brainscape_prep={root}",
            f"paths.brainscape_json={json_path}",
            "dataset.brainscape.num_workers=0",
            "dataset.brainscape.input_specs=[{key:latent,group:latent,modalities:[t1w, t2w]}]",
            "dataset.train_tf.image_keys=[]",
            "dataset.train_tf.passthrough_keys=[latent]",
            "dataset.val_tf.image_keys=[]",
            "dataset.val_tf.passthrough_keys=[latent]",
            "dataset.train_tf.meta_keys=[modality]",
            "dataset.brain_roi_size=[32,32,32]",
            "dataset.train_tf.patch_size=[32,32,32]",
        ]

        cfg = compose(config_name="dataset/brainscape_test", overrides=overrides)
        print(OmegaConf.to_yaml(cfg, resolve=True))
        
        dm = instantiate(cfg.dataset.brainscape)
        
        loaders = {
            "train": dm.train_dataloader(),
            "val": dm.val_dataloader(),
            "test": dm.test_dataloader(),
        }
        
        mods = set()
        for loader in loaders.values():
            for batch in loader:
                mods.add(batch["modality"][0])
                assert torch.all((batch["latent"] >= 0) & (batch["latent"] <= 1))
                assert batch["latent"].shape[-3:] == (16, 16, 16)
        assert mods == {"t1w", "t2w"}
        
         
        assert {"latent", "modality_map"}.issubset(batch.keys())
        assert torch.equal(batch["modality_map"], torch.tensor([0]))
        assert batch["latent"].shape[-3:] == (16, 16, 16)



def test_brainscape_latent_and_image(tmp_path):
    root, json_path = _generate_dummy_latent_dataset(tmp_path)
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            # Dataset Overrides
            f"paths.brainscape_prep={root}",
            f"paths.brainscape_json={json_path}",
            "dataset.brainscape.num_workers=0",
            "dataset.brainscape.input_specs=[{key:image,group:preprocessed,modalities:[t1w,t2w]},{key:latent,group:latent,modalities:[t1w,t2w]}]",
            "dataset.train_tf.image_keys=[image]",
            "dataset.train_tf.passthrough_keys=[latent]",
            "dataset.train_tf.meta_keys=[modality]",
            "dataset.brain_roi_size=[32,32,32]",
            "dataset.val_tf.image_keys=[image]",
            "dataset.val_tf.passthrough_keys=[latent]",
            "dataset.val_tf.meta_keys=[modality]",
            "dataset.conditioning.modality_conditioning.key_name=class_labels",
        ]
        cfg = compose(config_name="dataset/brainscape_test", overrides=overrides)
        dm = instantiate(cfg.dataset.brainscape)
        dm.setup()

        mods = set()
        for batch in dm.train_dataloader():
            mods.update(batch["modality"])
            assert {"image", "latent", "class_labels"}.issubset(batch.keys())
            img_shape = batch["image"].shape[-3:]
            lat_shape = batch["latent"].shape[-3:]
            assert all(i == l * 2 for i, l in zip(img_shape, lat_shape))
        assert mods == {"t1w", "t2w"}


latent_specs = "[{key:latent,group:latent,modalities:[t1w]}]"
latent_image_specs = (
    "[{key:image,group:preprocessed,modalities:[t1w,flair]},"
    "{key:latent,group:latent,modalities:[t1w,flair]}]"
)
latent_image_label_specs = (
    "[{key:image,group:preprocessed,modalities:[t1w,flair]},"
    "{key:label,group:preprocessed,modalities:[seg]},"
    "{key:latent,group:latent,modalities:[t1w,flair]}]"
)

@pytest.mark.parametrize("input_specs", [latent_specs, latent_image_specs, latent_image_label_specs])
def test_brainscape_input_spec_combinations(tmp_path, input_specs):
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        img_keys = "[image]" if "image" in input_specs else "[]"
        label_keys = "[label]" if "label" in input_specs else "[]"
        latent_keys = "[latent]" if "latent" in input_specs else "[]"
        overrides = [
            # Dataset Overrides
            "paths.brainscape_prep=${paths.masi_brainscape_prep_test}",
            "paths.brainscape_json=${paths.masi_brainscape_json_test}",
            "dataset.brainscape.num_workers=0",
            f"dataset.brainscape.input_specs={input_specs}",
            f"dataset.train_tf.image_keys={img_keys}",
            f"dataset.train_tf.label_keys={label_keys}",
            f"dataset.train_tf.passthrough_keys={latent_keys}",
            "dataset.train_tf.meta_keys=[modality]",
            "dataset.brain_roi_size=[64,64,64]",
            "dataset.conditioning.modality_conditioning.key_name=class_labels",
        ]
        cfg = compose(config_name="dataset/brainscape_test", overrides=overrides)
        print(OmegaConf.to_yaml(cfg, resolve=True))
        
        dm = instantiate(cfg.dataset.brainscape)
        dm.setup()

        expected_keys = {spec["key"] for spec in cfg.dataset.brainscape.input_specs}
        expected_mods: set[str] = set()
        for spec in cfg.dataset.brainscape.input_specs:
            expected_mods.update(spec.get("modalities", []))

        observed_mods: set[str] = set()

        for batch in dm.train_dataloader():
            observed_mods.update(batch["modality"])
            present_keys = {k for k in ["image", "label", "latent"] if k in batch}
            assert present_keys <= expected_keys

            if {"image", "latent"} <= present_keys:
                img_shape = batch["image"].shape[-3:]
                lat_shape = batch["latent"].shape[-3:]
                assert all(i > l  for i, l in zip(img_shape, lat_shape))
                assert torch.all((batch["latent"] >= 0) & (batch["latent"] <= 1))

            if "label" in present_keys and "image" in present_keys:
                assert batch["label"].shape[-3:] == batch["image"].shape[-3:]

        assert observed_mods.issubset(expected_mods)