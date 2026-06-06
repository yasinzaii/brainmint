import json
import sys
from math import isclose
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import nibabel as nib
import numpy as np
from omegaconf import OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from brainmint.data.transforms.demographics import DemographicsConditioningd


CONFIG_DIR = PROJECT_ROOT / "configs"


@pytest.fixture(scope="module")
def demographics_config():
    """Load and resolve the demographics configuration."""
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="dataset/demographics")
    return OmegaConf.to_container(cfg.dataset.conditioning.demographics_config, resolve=True)


@pytest.fixture()
def demographics_transform(demographics_config):
    return DemographicsConditioningd(config=demographics_config)


def _write_nifti(path: Path, shape=(4, 4, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.random.rand(*shape).astype("float32"), affine=np.eye(4))
    nib.save(img, path)


def _make_latent_json(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "latent_ds"
    json_path = tmp_path / "latent.json"

    shared_demo = {
        "age": 44,
        "height": 1.75,
        "weight": 80.0,
        "BMI": 26.1,
        "age_group": "40-45",
        "sex": "male",
        "race": "2",
        "handedness": "R",
        "education": "medium",
        "socio_economic": "high",
        "stroke": "no",
        "schizophrenia": "n/a",
        "depression": "no",
        "ADHD": "yes",
        "BIPOLAR": "no",
        "prosopagnosia": "n/a",
        "epilepsy": "no",
        "FCD": "n/a",
        "HS": "no",
        "tumor": "n/a",
        "acuteischaemicstroke": "no",
        "DNT": "n/a",
        "GL": "yes",
        "aneurysm": "no",
        "ASD": "n/a",
    }

    records = {
        split: [
            {
                "dataset": "demo",
                "subject": f"{split}_subject",
                "latent": {"t1w": f"{split}_latent.nii.gz"},
                "demographics": shared_demo,
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


def test_transform_outputs_expected_tensors(demographics_transform, demographics_config):
    ordered = demographics_config["ordered_fields"]
    fields = demographics_config["fields"]

    raw = {
        "age": 40,
        "height": None,  # missing numeric
        "weight": "72.919",  # numeric string
        "BMI": "n/a",  # explicit NA
        #"age_group": "35-40",
        "sex": "female",
        "race": "3",
        "handedness": "",  # empty treated as missing
        "education": "high",
        "socio_economic": "n/a",
        "stroke": "yes",
        "schizophrenia": "n/a",
        "depression": "no",
        "ADHD": "yes",
        "BIPOLAR": "no",
        "prosopagnosia": "n/a",
        "epilepsy": "no",
        "FCD": "yes",
        "HS": "no",
        "tumor": "n/a",
        "acuteischaemicstroke": "no",
        "DNT": "n/a",
        "GL": "yes",
        # "aneurysm": "no",  <= MISSING
        "ASD": "n/a",
    }

    out = demographics_transform({"demographics": raw})

    assert "demo_values" in out and "demo_missing" in out
    assert out["demo_values"].shape == (len(ordered),)
    assert out["demo_missing"].shape == (len(ordered),)
    assert out["demo_values"].dtype == torch.float32
    assert out["demo_missing"].dtype == torch.bool

    # Numeric normalization
    age_norm = (raw["age"] - fields["age"]["mean"]) / fields["age"]["std"]
    assert isclose(float(out["demo_dict"]["age"]), age_norm, rel_tol=1e-4)

    # Missing handling for numeric + categorical
    assert out["demo_missing"][ordered.index("height")]
    assert out["demo_missing"][ordered.index("BMI")]
    assert out["demo_missing"][ordered.index("handedness")]

    # Categorical mappings
    assert out["demo_dict"]["age_group"] == fields["age_group"]["mapping"]["40-45"]
    assert out["demo_dict"]["sex"] == fields["sex"]["mapping"]["female"]
    assert out["demo_dict"]["race"] == fields["race"]["mapping"]["3"]
    assert out["demo_dict"]["socio_economic"] == fields["socio_economic"]["mapping"]["n/a"]


def test_transform_ignores_missing_key_when_allowed(demographics_config):
    tfm = DemographicsConditioningd(config=demographics_config, allow_missing_keys=True)
    sample = {"not_demo": 123}
    assert tfm(sample) == sample


def test_transform_rejects_unknown_category(demographics_transform):
    with pytest.raises(ValueError):
        demographics_transform({"demographics": {"age": 10, "sex": "unknown"}})


@pytest.mark.parametrize("num_workers", [0])
def test_transform_runs_inside_brainscape_dataset(tmp_path, demographics_config, num_workers):
    root, json_path = _make_latent_json(tmp_path)

    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            f"dataset.brainscape.json_path={json_path}",
            f"dataset.brainscape.dataset_root={root}",
            f"dataset.brainscape.num_workers={num_workers}",
            "dataset.brainscape.subset_frac=null",
            "dataset.brainscape.batch_size=1",
            "dataset.brainscape.val_batch_size=1",
            "dataset.brainscape.test_batch_size=1",
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
            "dataset.train_tf.extra_xforms_end=[${dataset.conditioning.modality_conditioning},${dataset.conditioning.demographics_conditioning}]",
            "+dataset.val_tf.extra_xforms_end=[${dataset.conditioning.modality_conditioning},${dataset.conditioning.demographics_conditioning}]",
            "+dataset.test_tf.extra_xforms_end=[${dataset.conditioning.modality_conditioning},${dataset.conditioning.demographics_conditioning}]",
        ]

        cfg = compose(config_name="dataset/brainscape_test", overrides=overrides)
        demographics_cfg = OmegaConf.load(CONFIG_DIR / "dataset" / "demographics.yaml")

        with open_dict(cfg.dataset.conditioning):
            cfg.dataset.conditioning.demographics_config = demographics_cfg.demographics_config
            cfg.dataset.conditioning.demographics_conditioning = demographics_cfg.demographics_conditioning
            cfg.dataset.conditioning.demographics_conditioning.config = cfg.dataset.conditioning.demographics_config

        dm = instantiate(cfg.dataset.brainscape)
        dm.setup()

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert "latent" in batch
        assert "demo_values" in batch and "demo_missing" in batch
        assert batch["demo_values"].shape[1] == len(demographics_config["ordered_fields"])
        assert batch["demo_missing"].shape[1] == len(demographics_config["ordered_fields"])
        assert batch["demo_dict"]["sex"][0] == demographics_config["fields"]["sex"]["mapping"]["male"]

        # Ensure modality conditioning remains attached alongside demographics
        key_name = cfg.dataset.conditioning.modality_conditioning.key_name
        cfg_map = dict(cfg.dataset.conditioning.modality_map)
        assert key_name in batch
        assert isinstance(batch[key_name], torch.Tensor)

        for sample_idx in range(len(batch["latent"])):
            assert batch[key_name][sample_idx] in cfg_map.values()
            assert batch["modality"][sample_idx] in cfg_map
            assert batch["modality"][sample_idx] == list(cfg_map.keys())[list(cfg_map.values()).index(batch[key_name][sample_idx])]