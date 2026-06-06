import json
from pathlib import Path

import pytest
import torch

from brainmint.data.brainscape_paired import BrainScapePairedDataModule


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def _write_json(p: Path, payload: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))


def test_pair_filter_and_bucket_stats(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ds = "brats"

    # Real group files
    _touch(root / ds / "preprocessed" / "sub-001" / "t1w.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-001" / "flair.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-002" / "t1w.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-002" / "t2w.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-002" / "flair.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-003" / "t1w.nii.gz")

    # Synthetic group files (cached)
    _touch(root / ds / "preprocessed" / "sub-001" / "t2w_syn.nii.gz")

    train = [
        # s_miss: has t1w+flair in image, missing image t2w, but has synthetic t2w
        {
            "dataset": ds,
            "subject": "sub-001",
            "session": "ses-01",
            "preprocessed": {"t1w": "sub-001/t1w.nii.gz", "flair": "sub-001/flair.nii.gz"},
            "synthetic": {"t2w": "sub-001/t2w_syn.nii.gz"},
        },
        # s_real: has t1w+t2w+flair in image (synthetic may or may not exist)
        {
            "dataset": ds,
            "subject": "sub-002",
            "session": "ses-01",
            "preprocessed": {
                "t1w": "sub-002/t1w.nii.gz",
                "t2w": "sub-002/t2w.nii.gz",
                "flair": "sub-002/flair.nii.gz",
            },
        },
        # Invalid: missing flair => should be filtered by pair.require.all
        {
            "dataset": ds,
            "subject": "sub-003",
            "session": "ses-01",
            "preprocessed": {"t1w": "sub-003/t1w.nii.gz"},
        },
    ]

    jpath = tmp_path / "brainscape.json"
    _write_json(jpath, {"train": train, "val": train[:2], "test": train[:2]})

    dm = BrainScapePairedDataModule(
        json_path=jpath,
        dataset_root=root,
        modalities=["t1w", "t2w", "flair"],
        input_specs=[
            {"key": "image", "group": "preprocessed", "modalities": ["t1w", "t2w", "flair"], "subdir": "preprocessed"},
            {"key": "synthetic", "group": "synthetic", "modalities": ["t2w"], "subdir": "preprocessed"},
        ],
        pair={
            "emit": {"modalities": ["t1w", "t2w", "flair"], "keys": ["image", "synthetic"]},
            "require": {
                "all": {"image": ["t1w", "flair"]},
                "any": [{"keys": ["image", "synthetic"], "modalities": ["t2w"]}],
            },
        },
        buckets={
            "exhaustive": True,
            "exclusive": True,
            "rules": [
                {"name": "s_miss", "includes": {"image": ["t1w", "flair"], "synthetic": ["t2w"]}, "missing": {"image": ["t2w"]}},
                {"name": "s_real", "includes": {"image": ["t1w", "t2w", "flair"]}, "missing": {}},
            ],
        },
        use_bucket_sampler=False,
        num_workers=0,
        cache_rate=0.0,
    )

    dm.setup("fit")
    assert dm.get_bucket_stats("train") == {"s_miss": 1, "s_real": 1}

    batch = next(iter(dm.train_dataloader()))
    assert "image" in batch
    assert "synthetic" in batch
    assert "bucket" in batch


def test_bucket_unassigned_raises(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ds = "brats"

    _touch(root / ds / "preprocessed" / "sub-001" / "t1w.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-001" / "flair.nii.gz")

    train = [
        {
            "dataset": ds,
            "subject": "sub-001",
            "session": "ses-01",
            "preprocessed": {"t1w": "sub-001/t1w.nii.gz", "flair": "sub-001/flair.nii.gz"},
        }
    ]
    jpath = tmp_path / "brainscape.json"
    _write_json(jpath, {"train": train, "val": train, "test": train})

    dm = BrainScapePairedDataModule(
        json_path=jpath,
        dataset_root=root,
        modalities=["t1w", "flair"],
        input_specs=[{"key": "image", "group": "preprocessed", "modalities": ["t1w", "flair"]}],
        pair={"emit": {"modalities": ["t1w", "flair"], "keys": ["image"]}, "require": {"all": {"image": ["t1w", "flair"]}}},
        buckets={
            "exhaustive": True,
            "exclusive": True,
            "rules": [
                {"name": "never", "includes": {}, "missing": {"image": ["t1w"]}},
            ],
        },
        num_workers=0,
    )
    with pytest.raises(RuntimeError, match="did not match any bucket"):
        dm.setup("fit")


def test_bucket_overlap_raises(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ds = "brats"
    _touch(root / ds / "preprocessed" / "sub-001" / "t1w.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-001" / "flair.nii.gz")
    train = [
        {
            "dataset": ds,
            "subject": "sub-001",
            "session": "ses-01",
            "preprocessed": {"t1w": "sub-001/t1w.nii.gz", "flair": "sub-001/flair.nii.gz"},
        }
    ]
    jpath = tmp_path / "brainscape.json"
    _write_json(jpath, {"train": train, "val": train, "test": train})

    dm = BrainScapePairedDataModule(
        json_path=jpath,
        dataset_root=root,
        modalities=["t1w", "flair"],
        input_specs=[{"key": "image", "group": "preprocessed", "modalities": ["t1w", "flair"]}],
        pair={"emit": {"modalities": ["t1w", "flair"], "keys": ["image"]}},
        buckets={
            "exhaustive": True,
            "exclusive": True,
            "rules": [
                {"name": "a", "includes": {"image": ["t1w"]}},
                {"name": "b", "includes": {"image": ["t1w"]}},
            ],
        },
        num_workers=0,
    )
    with pytest.raises(RuntimeError, match="matches multiple buckets"):
        dm.setup("fit")


def test_emit_key_not_in_input_specs_raises(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ds = "brats"
    _touch(root / ds / "preprocessed" / "sub-001" / "t1w.nii.gz")

    train = [
        {"dataset": ds, "subject": "sub-001", "session": "ses-01", "preprocessed": {"t1w": "sub-001/t1w.nii.gz"}}
    ]
    jpath = tmp_path / "brainscape.json"
    _write_json(jpath, {"train": train, "val": train, "test": train})

    with pytest.raises(ValueError, match="pair\\.emit\\.keys"):
        BrainScapePairedDataModule(
            json_path=jpath,
            dataset_root=root,
            modalities=["t1w"],
            input_specs=[{"key": "image", "group": "preprocessed", "modalities": ["t1w"]}],
            pair={"emit": {"modalities": ["t1w"], "keys": ["image", "synthetic"]}},
            buckets={"rules": [{"name": "any", "includes": {"image": ["t1w"]}, "missing": {}}]},
            num_workers=0,
        )


def test_bucket_prob_update_changes_weights(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ds = "brats"
    _touch(root / ds / "preprocessed" / "sub-001" / "t1w.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-001" / "flair.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-001" / "t2w_syn.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-002" / "t1w.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-002" / "t2w.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-002" / "flair.nii.gz")

    train = [
        {
            "dataset": ds,
            "subject": "sub-001",
            "session": "ses-01",
            "preprocessed": {"t1w": "sub-001/t1w.nii.gz", "flair": "sub-001/flair.nii.gz"},
            "synthetic": {"t2w": "sub-001/t2w_syn.nii.gz"},
        },
        {
            "dataset": ds,
            "subject": "sub-002",
            "session": "ses-01",
            "preprocessed": {"t1w": "sub-002/t1w.nii.gz", "t2w": "sub-002/t2w.nii.gz", "flair": "sub-002/flair.nii.gz"},
        },
    ]
    jpath = tmp_path / "brainscape.json"
    _write_json(jpath, {"train": train, "val": train, "test": train})

    dm = BrainScapePairedDataModule(
        json_path=jpath,
        dataset_root=root,
        modalities=["t1w", "t2w", "flair"],
        input_specs=[
            {"key": "image", "group": "preprocessed", "modalities": ["t1w", "t2w", "flair"], "subdir": "preprocessed"},
            {"key": "synthetic", "group": "synthetic", "modalities": ["t2w"], "subdir": "preprocessed"},
        ],
        pair={
            "emit": {"modalities": ["t1w", "t2w", "flair"], "keys": ["image", "synthetic"]},
            "require": {
                "all": {"image": ["t1w", "flair"]},
                "any": [{"keys": ["image", "synthetic"], "modalities": ["t2w"]}],
            },
        },
        buckets={
            "exhaustive": True,
            "exclusive": True,
            "rules": [
                {"name": "s_miss", "includes": {"image": ["t1w", "flair"], "synthetic": ["t2w"]}, "missing": {"image": ["t2w"]}},
                {"name": "s_real", "includes": {"image": ["t1w", "t2w", "flair"]}, "missing": {}},
            ],
        },
        use_bucket_sampler=True,
        num_workers=0,
        cache_rate=0.0,
    )
    dm.setup("fit")
    assert dm._train_sampler is not None
    old = dm._train_sampler.weights.clone()

    # Force all mass to s_real (should zero out s_miss weights)
    dm.set_bucket_probs({"s_real": 1.0, "s_miss": 0.0})
    new = dm._train_sampler.weights
    assert float(new.sum()) > 0
    assert not torch.allclose(old, new)


def test_set_bucket_probs_validation(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ds = "brats"
    _touch(root / ds / "preprocessed" / "sub-001" / "t1w.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-001" / "flair.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-001" / "t2w_syn.nii.gz")
    train = [
        {
            "dataset": ds,
            "subject": "sub-001",
            "session": "ses-01",
            "preprocessed": {"t1w": "sub-001/t1w.nii.gz", "flair": "sub-001/flair.nii.gz"},
            "synthetic": {"t2w": "sub-001/t2w_syn.nii.gz"},
        }
    ]
    jpath = tmp_path / "brainscape.json"
    _write_json(jpath, {"train": train, "val": train, "test": train})

    dm = BrainScapePairedDataModule(
        json_path=jpath,
        dataset_root=root,
        modalities=["t1w", "t2w", "flair"],
        input_specs=[
            {"key": "image", "group": "preprocessed", "modalities": ["t1w", "t2w", "flair"], "subdir": "preprocessed"},
            {"key": "synthetic", "group": "synthetic", "modalities": ["t2w"], "subdir": "preprocessed"},
        ],
        pair={
            "emit": {"modalities": ["t1w", "t2w", "flair"], "keys": ["image", "synthetic"]},
            "require": {"all": {"image": ["t1w", "flair"]}, "any": [{"keys": ["image", "synthetic"], "modalities": ["t2w"]}]},
        },
        buckets={"rules": [{"name": "s_miss", "includes": {"image": ["t1w", "flair"], "synthetic": ["t2w"]}, "missing": {"image": ["t2w"]}}]},
        use_bucket_sampler=True,
        num_workers=0,
        cache_rate=0.0,
    )
    dm.setup("fit")

    with pytest.raises(ValueError):
        dm.set_bucket_probs({"unknown": 1.0})
    with pytest.raises(ValueError):
        dm.set_bucket_probs({"s_miss": -1.0})
    with pytest.raises(ValueError):
        dm.set_bucket_probs({"s_miss": 0.0})


def test_seg_map_allowed_and_bucket_rule_can_reference_it(tmp_path: Path) -> None:
    ds = "brats"
    root = tmp_path / "root"

    # Create files (image + seg mask)
    _touch(root / ds / "preprocessed" / "sub-001" / "t1w.nii.gz")
    _touch(root / ds / "preprocessed" / "sub-001" / "seg.nii.gz")

    train = [
        {
            "dataset": ds,
            "subject": "sub-001",
            "session": "ses-01",
            "preprocessed": {"t1w": "sub-001/t1w.nii.gz", "seg": "sub-001/seg.nii.gz"},
        }
    ]
    jpath = tmp_path / "brainscape.json"
    _write_json(jpath, {"train": train, "val": train, "test": train})

    dm = BrainScapePairedDataModule(
        json_path=jpath,
        dataset_root=root,
        modalities=["t1w", "t2w"],  # seg is intentionally excluded
        input_specs=[
            {"key": "image", "group": "preprocessed", "modalities": ["t1w"]},
            {"key": "seg", "group": "preprocessed", "modalities": ["seg"], "seg_map": True},
        ],
        pair={"emit": {"modalities": ["t1w"], "keys": ["image", "seg"]}},
        buckets={
            "exhaustive": True,
            "exclusive": True,
            "rules": [
                {"name": "has_seg", "includes": {"seg": ["seg"]}, "missing": {}},
            ],
        },
        num_workers=0,
    )
    dm.setup("fit")
    assert dm.get_bucket_stats("train") == {"has_seg": 1}
