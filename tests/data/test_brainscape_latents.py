import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
TEST_CONFIG_DIR = PROJECT_ROOT / "tests" / "fixtures" / "configs"

from tests.helpers.brainscape import make_minimal_brainscape_dataset  # noqa: E402


def _generate_dummy_latent_dataset(tmp_path: Path) -> tuple[Path, Path]:
    fixture = Path("tests/fixtures/configs/data/minimal_brainscape.json")
    dataset_json = make_minimal_brainscape_dataset(
        tmp_path,
        fixture,
        image_size=16,
    )
    return tmp_path, dataset_json


def _compose_minimal_brainscape(tmp_path, monkeypatch, overrides):
    root, json_path = _generate_dummy_latent_dataset(tmp_path)
    monkeypatch.setenv("BRAINMINT_TEST_DATASET_ROOT", str(root))
    monkeypatch.setenv("BRAINMINT_TEST_BRAINSCAPE_JSON", str(json_path))

    with initialize_config_dir(str(TEST_CONFIG_DIR), version_base=None):
        return compose(config_name="data/minimal_brainscape", overrides=overrides)


def test_brainscape_latent_only(tmp_path, monkeypatch):
    cfg = _compose_minimal_brainscape(
        tmp_path,
        monkeypatch,
        overrides=[
            "dataset.brainscape.modalities=[t1w,t2w]",
            "dataset.brainscape.input_specs=[{key:latent,group:preprocessed,modalities:[t1w,t2w]}]",
            "dataset.train_tf.image_keys=[]",
            "dataset.train_tf.passthrough_keys=[latent]",
            "dataset.val_tf.image_keys=[]",
            "dataset.val_tf.passthrough_keys=[latent]",
        ],
    )
    print(OmegaConf.to_yaml(cfg, resolve=True))

    dm = instantiate(cfg.dataset.brainscape)
    loaders = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test": dm.test_dataloader(),
    }

    mods = set()
    batch = None
    for loader in loaders.values():
        for batch in loader:
            mods.update(batch["modality"])
            assert torch.all((batch["latent"] >= 0) & (batch["latent"] <= 1))
            assert batch["latent"].shape[-3:] == (16, 16, 16)
    assert mods == {"t1w", "t2w"}

    assert batch is not None
    assert {"latent", "modality_map"}.issubset(batch.keys())
    assert set(batch["modality_map"].flatten().tolist()) <= {0, 1}
    assert batch["latent"].shape[-3:] == (16, 16, 16)


def test_brainscape_latent_and_image(tmp_path, monkeypatch):
    cfg = _compose_minimal_brainscape(
        tmp_path,
        monkeypatch,
        overrides=[
            "dataset.brainscape.modalities=[t1w,t2w]",
            "dataset.brainscape.input_specs=[{key:image,group:preprocessed,modalities:[t1w,t2w]},{key:latent,group:preprocessed,modalities:[t1w,t2w]}]",
            "dataset.train_tf.passthrough_keys=[latent]",
            "dataset.val_tf.passthrough_keys=[latent]",
            "dataset.conditioning.modality_conditioning.key_name=class_labels",
        ],
    )
    dm = instantiate(cfg.dataset.brainscape)
    dm.setup()

    mods = set()
    for batch in dm.train_dataloader():
        mods.update(batch["modality"])
        assert {"image", "latent", "class_labels"}.issubset(batch.keys())
        img_shape = batch["image"].shape[-3:]
        lat_shape = batch["latent"].shape[-3:]
        assert img_shape == lat_shape == (16, 16, 16)
    assert mods == {"t1w", "t2w"}


latent_specs = "[{key:latent,group:preprocessed,modalities:[t1w]}]"
latent_image_specs = (
    "[{key:image,group:preprocessed,modalities:[t1w,flair]},"
    "{key:latent,group:preprocessed,modalities:[t1w,flair]}]"
)
latent_image_label_specs = (
    "[{key:image,group:preprocessed,modalities:[t1w,flair]},"
    "{key:label,group:preprocessed,modalities:[seg],seg_map:true},"
    "{key:latent,group:preprocessed,modalities:[t1w,flair]}]"
)


@pytest.mark.parametrize("input_specs", [latent_specs, latent_image_specs, latent_image_label_specs])
def test_brainscape_input_spec_combinations(tmp_path, monkeypatch, input_specs):
    img_keys = "[image]" if "image" in input_specs else "[]"
    label_keys = "[label]" if "label" in input_specs else "[]"
    latent_keys = "[latent]" if "latent" in input_specs else "[]"
    cfg = _compose_minimal_brainscape(
        tmp_path,
        monkeypatch,
        overrides=[
            "dataset.brainscape.modalities=[t1w,flair]",
            f"dataset.brainscape.input_specs={input_specs}",
            f"dataset.train_tf.image_keys={img_keys}",
            f"dataset.train_tf.label_keys={label_keys}",
            f"dataset.train_tf.passthrough_keys={latent_keys}",
            "dataset.conditioning.modality_conditioning.key_name=class_labels",
        ],
    )
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
            assert img_shape == lat_shape == (16, 16, 16)
            assert torch.all((batch["latent"] >= 0) & (batch["latent"] <= 1))

        if "label" in present_keys and "image" in present_keys:
            assert batch["label"].shape[-3:] == batch["image"].shape[-3:]

    assert observed_mods.issubset(expected_mods)
