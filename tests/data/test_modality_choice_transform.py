from __future__ import annotations

import pytest

import torch

from brainmint.data.transforms.modality_choice import ChooseStreamForModalitiesd, SharedChoiceState


ALL_MODALITIES = ["t1w", "t2w", "flair", "t1ce"]


def test_choose_stream_with_probs() -> None:
    # Two sources for t2w. Selection is stochastic but constrained by choices.
    tf = ChooseStreamForModalitiesd(
        modalities=["t1w", "t2w"],
        all_modalities=ALL_MODALITIES,
        choices={
            "s_real": {
                "t2w": {
                    "streams": {"image": "image", "synthetic": "synthetic"},
                    "probs": {"image": 0.7, "synthetic": 0.3},
                },
                "*": {"streams": {"image": "image"}, "probs": {"image": 1.0}},
            }
        },
        synthetic_stream_keys=["synthetic"],
        drop_bucket=False,
        out_key_map={"image": "image", "synthetic": "image"},
    )

    sample = {
        "record_id": "ds|sub|ses|||",
        "bucket": "s_real",
        "image": {"t1w": "t1_real.nii.gz", "t2w": "t2_real.nii.gz"},
        "synthetic": {"t2w": "t2_syn.nii.gz"},
    }

    out = tf(sample)
    assert out["image"]["t1w"] == "t1_real.nii.gz"
    assert out["image"]["t2w"] in ("t2_real.nii.gz", "t2_syn.nii.gz")
    assert out["is_mod_synthetic"].shape == (4,)
    assert out["bucket"] == "s_real"


def test_missing_modality_raises() -> None:
    tf = ChooseStreamForModalitiesd(
        modalities=["t2w"],
        all_modalities=ALL_MODALITIES,
        choices={"b": {"t2w": {"streams": {"image": "image"}, "probs": {"image": 1.0}}}},
        out_key_map={"image": "image"},
    )
    sample = {"record_id": "x", "bucket": "b", "image": {"t1w": "t1.nii.gz"}}
    with pytest.raises(KeyError, match="t2w"):
        tf(sample)


def test_multi_source_requires_probs_when_enabled() -> None:
    tf = ChooseStreamForModalitiesd(
        modalities=["t2w"],
        all_modalities=ALL_MODALITIES,
        choices={"b": {"t2w": {"streams": {"image": "image", "synthetic": "synthetic"}, "probs": {}}}},
        out_key_map={"image": "image", "synthetic": "image"},
    )
    sample = {
        "record_id": "x",
        "bucket": "b",
        "image": {"t2w": "t2_real.nii.gz"},
        "synthetic": {"t2w": "t2_syn.nii.gz"},
    }
    with pytest.raises(ValueError, match="sum of weights"):
        tf(sample)


def test_state_updates_are_observed() -> None:
    state = SharedChoiceState(
        choices={"s_real": {"t2w": {"streams": {"image": "image"}, "probs": {"image": 1.0}}}},
        epoch=0,
        seed=0,
        shared=False,
    )

    tf = ChooseStreamForModalitiesd(
        modalities=["t2w"],
        all_modalities=ALL_MODALITIES,
        state=state,
        out_key_map={"image": "image", "synthetic": "image"},
    )

    sample = {
        "record_id": "ds|sub|ses|||",
        "bucket": "s_real",
        "image": {"t2w": "t2_real.nii.gz"},
        "synthetic": {"t2w": "t2_syn.nii.gz"},
    }

    assert tf(sample)["image"]["t2w"] == "t2_real.nii.gz"

    state.set_choices({"s_real": {"t2w": {"streams": {"synthetic": "synthetic"}, "probs": {"synthetic": 1.0}}}})
    assert tf(sample)["image"]["t2w"] == "t2_syn.nii.gz"


def test_outputs_synthetic_mask_correctly() -> None:
    tf = ChooseStreamForModalitiesd(
        modalities=["t1w", "t2w"],
        all_modalities=ALL_MODALITIES,
        choices={
            "b": {
                "t2w": {"streams": {"synthetic": "synthetic"}, "probs": {"synthetic": 1.0}},
                "*": {"streams": {"image": "image"}, "probs": {"image": 1.0}},
            }
        },
        synthetic_stream_keys=["synthetic"],
        drop_bucket=True,
        out_key_map={"image": "image", "synthetic": "image"},
    )

    sample = {
        "record_id": "id",
        "bucket": "b",
        "image": {"t1w": "t1.nii.gz", "t2w": "t2_real.nii.gz"},
        "synthetic": {"t2w": "t2_syn.nii.gz"},
    }

    out = tf(sample)
    assert out["is_mod_synthetic"].tolist() == [0, 1, 0, 0]
    assert "bucket" not in out


def test_stream_list_carries_sigma_when_available() -> None:
    tf = ChooseStreamForModalitiesd(
        modalities=["t1w"],
        all_modalities=ALL_MODALITIES,
        choices={"b": {"t1w": {"streams": {"latent": ["latent", "latent_sigma"]}, "probs": {"latent": 1.0}}}},
        out_key_map={"latent": "latent", "latent_sigma": "latent_sigma"},
    )

    z_mu = torch.zeros(1, 2, 2, 2)
    z_sigma = torch.ones(1, 2, 2, 2)
    sample = {
        "record_id": "id",
        "bucket": "b",
        "latent": {"t1w": z_mu},
        "latent_sigma": {"t1w": z_sigma},
    }

    out = tf(sample)
    assert torch.allclose(out["latent"]["t1w"], z_mu)
    assert torch.allclose(out["latent_sigma"]["t1w"], z_sigma)


def test_single_stream_without_sigma_falls_back() -> None:
    tf = ChooseStreamForModalitiesd(
        modalities=["t1w"],
        all_modalities=ALL_MODALITIES,
        choices={"b": {"t1w": {"streams": {"latent": "latent"}, "probs": {"latent": 1.0}}}},
        out_key_map={"latent": "latent"},
    )

    z_mu = torch.zeros(1, 2, 2, 2)
    sample = {
        "record_id": "id",
        "bucket": "b",
        "latent": {"t1w": z_mu},
    }

    out = tf(sample)
    assert torch.allclose(out["latent"]["t1w"], z_mu)
