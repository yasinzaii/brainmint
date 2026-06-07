# tests/data/test_brainscape.py
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR   = PROJECT_ROOT / "configs"
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


@pytest.mark.parametrize("num_workers", [0])
def test_dataset_builds_and_loads(tmp_path, monkeypatch, num_workers):

    # Overrides
    overrides=[
            f"dataset.brainscape.num_workers={num_workers}",
    ]

    # Composing 
    cfg = _compose_minimal_brainscape(tmp_path, monkeypatch, overrides=overrides)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    dm = instantiate(cfg.dataset.brainscape)    # BrainScapeDataModule
    dm.setup()                                  # builds datasets

    # 1) Loader sanity: first batch must contain 'image'
    loaders = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test": dm.test_dataloader(),
    }

    for loader in loaders.values():
        batch = next(iter(loader))
        assert "image" in batch, "Missing 'image' key in batch dict"

    # 2) Printing Dataset/Dataloader Split Name and Legths 
    msg = ""
    for k, v in dm.get_splits_length().items():
        msg += f" {k}: {v},"    
    print(f"Dataset Splits >> {msg}")
    print("Batch Sizes >> " + ", ".join(f"{split}: {len(next(iter(loader))['image'])}" for split, loader in loaders.items()))
    print("Trainloader Sizes >> " + ", ".join(f"{split}: {len(loader)}" for split, loader in loaders.items()))

    # 3) Deterministic check
    loader1 = dm.val_dataloader()
    loader2 = dm.val_dataloader()
    first1  = next(iter(loader1))["image"][0].sum()
    first2  = next(iter(loader2))["image"][0].sum()
    assert torch.equal(first1, first2), "Shuffling not reproducible"


def test_modality_class_labels_and_mapping(tmp_path, monkeypatch, num_workers=0):
    overrides=[
        f"dataset.brainscape.num_workers={num_workers}",
    ]

    cfg = _compose_minimal_brainscape(tmp_path, monkeypatch, overrides=overrides)

    dm = instantiate(cfg.dataset.brainscape)
    dm.setup()

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    assert "image" in batch, "Missing 'image' key in batch"

    # key_name should be present
    key_name = cfg.dataset.conditioning.modality_conditioning.key_name
    assert key_name in batch, "key_name missing; transform not attached?"
    assert isinstance(batch[key_name], torch.Tensor)

    cfg_map = dict(cfg.dataset.conditioning.modality_map)

    for batch in train_loader:
        for idx in range(len(batch["image"])):
            assert batch[key_name][idx] in cfg_map.values(), f"Batch Image {key_name}:{batch[key_name][idx]} is not present in Target Mapping:{cfg_map}"
            assert batch["modality"][idx] in cfg_map.keys(), f"Batch Image Modality:{batch['modality'][idx]} is not present in Target Mapping:{cfg_map}"
            key_by_val = list(cfg_map.keys())[list(cfg_map.keys()).index(batch["modality"][idx])]  # Return Key from cfg_map by val: batch["modality"][idx]
            assert batch["modality"][idx] == key_by_val, f"Batch Image Modality:{batch['modality'][idx]} is not matching Target Mapping Modality:{key_by_val}"
