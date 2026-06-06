from __future__ import annotations

import pytest

import torch

from brainmint.data.transforms.modality_choice import ChooseStreamForModalitiesd, SharedChoiceState, SharedSamplingState


def test_choose_stream_with_probs() -> None:
    # Two sources for t2w. Deterministic selection is driven by record_id.
    tf = ChooseStreamForModalitiesd(
        modalities=["t1w", "t2w"],
        stream_keys=["image", "synthetic"],
        probs={
            "s_real": {
                "t2w": {"image": 0.7, "synthetic": 0.3},
                "*": {"image": 1.0},
            }
        },
        deterministic=False,
        synthetic_streams=["synthetic"],
        require_probs_for_multi_source=True,
        drop_bucket=False,
    )

    sample = {
        "record_id": "ds|sub|ses|||",
        "bucket": "s_real",
        "image": {"t1w": "t1_real.nii.gz", "t2w": "t2_real.nii.gz"},
        "synthetic": {"t2w": "t2_syn.nii.gz"},
    }

    out = tf(sample)
    assert out["t1w"] == "t1_real.nii.gz"
    assert out["t2w"] in ("t2_real.nii.gz", "t2_syn.nii.gz")
    assert out["chosen_stream_ids"].shape == (2,)
    assert out["chosen_is_synthetic"].shape == (2,)
    assert out["bucket"] == "s_real"


def test_missing_modality_raises() -> None:
    tf = ChooseStreamForModalitiesd(
        modalities=["t2w"],
        stream_keys=["image", "synthetic"],
        probs={"b": {"t2w": {"image": 1.0}}},
        deterministic=False,
        require_probs_for_multi_source=True,
    )
    sample = {"record_id": "x", "bucket": "b", "image": {"t1w": "t1.nii.gz"}}
    with pytest.raises(KeyError, match="Missing modality"):
        tf(sample)


def test_multi_source_requires_probs_when_enabled() -> None:
    tf = ChooseStreamForModalitiesd(
        modalities=["t2w"],
        stream_keys=["image", "synthetic"],
        probs={},  # empty
        deterministic=False,
        require_probs_for_multi_source=True,
    )
    sample = {
        "record_id": "x",
        "bucket": "b",
        "image": {"t2w": "t2_real.nii.gz"},
        "synthetic": {"t2w": "t2_syn.nii.gz"},
    }
    with pytest.raises(ValueError, match="No probs defined"):
        tf(sample)


def test_state_updates_are_observed() -> None:
    state = SharedChoiceState(probs={"s_real": {"t2w": {"image": 1.0}}}, epoch=0, seed=0, shared=False)

    tf = ChooseStreamForModalitiesd(
        modalities=["t2w"],
        stream_keys=["image", "synthetic"],
        state=state,
        deterministic=False,
        require_probs_for_multi_source=True,
    )

    sample = {
        "record_id": "ds|sub|ses|||",
        "bucket": "s_real",
        "image": {"t2w": "t2_real.nii.gz"},
        "synthetic": {"t2w": "t2_syn.nii.gz"},
    }

    assert tf(sample)["t2w"] == "t2_real.nii.gz"

    state.set_probs({"s_real": {"t2w": {"synthetic": 1.0}}})
    assert tf(sample)["t2w"] == "t2_syn.nii.gz"


def test_outputs_synthetic_mask_correctly() -> None:
    tf = ChooseStreamForModalitiesd(
        modalities=["t1w", "t2w"],
        stream_keys=["image", "synthetic"],
        probs={"b": {"t2w": {"synthetic": 1.0}, "*": {"image": 1.0}}},
        deterministic=False,
        synthetic_streams=["synthetic"],
        require_probs_for_multi_source=True,
        drop_bucket=True,
        modalities_all=["t1w", "t2w", "flair", "t1ce"],
    )

    sample = {
        "record_id": "id",
        "bucket": "b",
        "image": {"t1w": "t1.nii.gz", "t2w": "t2_real.nii.gz"},
        "synthetic": {"t2w": "t2_syn.nii.gz"},
    }

    out = tf(sample)
    assert out["chosen_is_synthetic"].tolist() == [0, 1]
    assert out["chosen_is_synthetic_full"].tolist() == [0, 1, 0, 0]
    assert "bucket" not in out


def test_partial_sampling_applies_when_sigma_available() -> None:
    torch.manual_seed(0)
    state = SharedSamplingState(config={"sigma_prob": 1.0, "sigma_alpha": 1.0}, shared=False)
    tf = ChooseStreamForModalitiesd(
        modalities=["t1w"],
        stream_keys=["latent"],
        probs={},
        deterministic=False,
        require_probs_for_multi_source=False,
        sample_latents=True,
        sigma_pairing={"latent": "latent_sigma"},
        sampling_state=state,
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
    assert not torch.allclose(out["t1w"], z_mu)


def test_partial_sampling_missing_sigma_falls_back() -> None:
    tf = ChooseStreamForModalitiesd(
        modalities=["t1w"],
        stream_keys=["latent"],
        probs={},
        deterministic=False,
        require_probs_for_multi_source=False,
        sample_latents=True,
        sigma_pairing={"latent": "latent_sigma"},
        require_sigma_for_sampling=False,
    )

    z_mu = torch.zeros(1, 2, 2, 2)
    sample = {
        "record_id": "id",
        "bucket": "b",
        "latent": {"t1w": z_mu},
    }

    out = tf(sample)
    assert torch.allclose(out["t1w"], z_mu)
