import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from brainmint.data.transforms.demographics import DemographicsConditioningd  # noqa: E402
from brainmint.models.conditioning.demographics_encoder import DemographicsEncoder  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "configs"


@pytest.fixture(scope="module")
def demographics_config():
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="dataset/demographics")
    return OmegaConf.to_container(cfg.dataset.conditioning.demographics_config, resolve=True)


@pytest.fixture()
def encoder(demographics_config):
    return DemographicsEncoder(config=demographics_config, dem_embed_dim=16)


@pytest.fixture()
def transform(demographics_config):
    return DemographicsConditioningd(config=demographics_config)


def _write_nifti(path, shape=(4, 4, 4)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.random.rand(*shape).astype("float32"), affine=np.eye(4))
    nib.save(img, path)


def _make_latent_json(tmp_path: Path, demo_cfg: dict) -> tuple[Path, Path]:
    root = tmp_path / "latent_ds"
    json_path = tmp_path / "latent.json"

    records = {
        split: [
            {
                "dataset": "demo",
                "subject": f"{split}_subject",
                "latent": {"t1w": f"{split}_latent.nii.gz"},
                "demographics": {name: "n/a" for name in demo_cfg["ordered_fields"]},
            }
        ]
        for split in ["train", "val", "test"]
    }

    for split in records.values():
        for rec in split:
            for rel_path in rec["latent"].values():
                _write_nifti(root / rec["dataset"] / "preprocessed" / rel_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(records, f)
    return root, json_path


def test_encoder_forward_produces_embedding(transform, encoder, demographics_config):
    sample = {"demographics": {name: "n/a" for name in demographics_config["ordered_fields"]}}
    out = transform(sample)

    demo_values = out["demo_values"].unsqueeze(0)
    demo_missing = out["demo_missing"].unsqueeze(0)

    emb = encoder(demo_values, demo_missing)
    assert emb.shape == (1, 16)
    assert emb.dtype == torch.float32
    assert emb.requires_grad


def test_encoder_rejects_noncontiguous_mappings(demographics_config):
    bad_cfg = OmegaConf.create(demographics_config)
    bad_cfg.fields["sex"]["mapping"] = {"n/a": 0, "female": 2}

    with pytest.raises(ValueError):
        DemographicsEncoder(config=bad_cfg, dem_embed_dim=4)


def test_encoder_handles_dataset_batch(tmp_path, demographics_config, encoder):
    root, json_path = _make_latent_json(tmp_path, demographics_config)

    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            f"dataset.brainscape.json_path={json_path}",
            f"dataset.brainscape.dataset_root={root}",
            "dataset.brainscape.num_workers=0",
            "dataset.brainscape.batch_size=1",
            "dataset.brainscape.val_batch_size=1",
            "dataset.brainscape.test_batch_size=1",
            "dataset.brainscape.subset_frac=null",
            "dataset.brainscape.modalities=[t1w]",
            "dataset.brainscape.input_specs=[{key:latent,group:latent,modalities:[t1w]}]",
            "dataset.train_tf._target_=brainmint.data.transforms.mri_vae.VAETransform",
            "dataset.train_tf.is_train=false",
            "dataset.train_tf.random_aug=false",
            "dataset.train_tf.image_keys=[]",
            "dataset.train_tf.passthrough_keys=[latent]",
            "dataset.train_tf.meta_keys=[modality,dataset,subject,demographics]",
            "dataset.train_tf.brain_roi_size=[4,4,4]",
            "dataset.train_tf.patch_size=[4,4,4]",
            "+dataset.train_tf.val_patch_size=null",
            "dataset.val_tf=${dataset.train_tf}",
            "dataset.test_tf=${dataset.train_tf}",
        ]

        cfg = compose(config_name="dataset/brainscape_test", overrides=overrides)
        demographics_cfg = OmegaConf.load(CONFIG_DIR / "dataset" / "demographics.yaml")

        with open_dict(cfg.dataset.conditioning):
            cfg.dataset.conditioning.demographics_config = demographics_cfg.demographics_config
            cfg.dataset.conditioning.demographics_conditioning = demographics_cfg.demographics_conditioning
            cfg.dataset.conditioning.demographics_conditioning.config = cfg.dataset.conditioning.demographics_config

        for split_tf in ("train_tf", "val_tf", "test_tf"):
            with open_dict(cfg.dataset):
                cfg.dataset[split_tf].extra_xforms_end = [
                    cfg.dataset.conditioning.modality_conditioning,
                    cfg.dataset.conditioning.demographics_conditioning,
                ]

        dm = instantiate(cfg.dataset.brainscape)
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        demo_values = batch["demo_values"]
        demo_missing = batch["demo_missing"]

        emb = encoder(demo_values, demo_missing)
        assert emb.shape == (1, 16)
        assert emb.requires_grad
