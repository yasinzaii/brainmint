import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "tests" / "fixtures" / "configs"


def build_simple_tf():
    import torch
    from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, Lambdad, LoadImaged

    from brainmint.data.transforms.conditioning import MapModalityToLabeld
    return Compose([
        LoadImaged(keys=["image", "latent"]),
        EnsureChannelFirstd(keys=["image", "latent"]),
        EnsureTyped(keys=["image", "latent"], dtype=torch.float32),
        Lambdad(keys="latent", func=lambda x: torch.as_tensor(x).clamp_(0, 1)),
        MapModalityToLabeld(mapping={"t1w": 0}, key_name="class_labels"),
    ])


def _make_json(tmp_path: Path, with_image: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "ds_root"
    (root / "ds1" / "preprocessed").mkdir(parents=True)

    lat = np.random.rand(1, 4, 4, 4).astype("float32")
    np.save(root / "ds1" / "preprocessed" / "lat.npy", lat)

    rec = {"dataset": "ds1", "latent": {"t1w": "lat.npy"}, "subject": "s1"}
    if with_image:
        import nibabel as nib
        img = np.random.rand(4, 4, 4).astype("float32")
        nif = nib.Nifti1Image(img, affine=np.eye(4))
        nib.save(nif, root / "ds1" / "preprocessed" / "img.nii.gz")
        rec["preprocessed"] = {"t1w": "img.nii.gz"}
    data = {"train": [rec], "val": [rec], "test": [rec]}
    json_path = tmp_path / "data.json"
    with json_path.open("w") as f:
        json.dump(data, f)
    return root, json_path


@pytest.mark.parametrize("with_conditioning,assert_keys", [
    (False, {"latent"}),
    (True, {"latent", "class_labels"}),
])
def test_brainscape_latent_variants(tmp_path, with_conditioning, assert_keys):
    """Latent-only dataset with optional conditioning."""
    root, json_path = _make_json(tmp_path)
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            f"dataset.brainscape.json_path={json_path}",
            f"dataset.brainscape.dataset_root={root}",
            "dataset.brainscape.num_workers=0",
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
            "dataset.train_tf.meta_keys=[modality]",
            "dataset.train_tf.brain_roi_size=[4,4,4]",
            "dataset.train_tf.patch_size=[4,4,4]",
            "+dataset.train_tf.val_patch_size=null",
            "dataset.val_tf=${dataset.train_tf}",
            "dataset.test_tf=${dataset.train_tf}",
        ]
        if with_conditioning:
            overrides.append(
                "dataset.train_tf.extra_xforms_end=[${dataset.conditioning.modality_conditioning}]"
            )
            overrides.append(
                "dataset.conditioning.modality_conditioning.key_name=class_labels"
            )
        cfg = compose(config_name="data/minimal_brainscape", overrides=overrides)
        dm = instantiate(cfg.dataset.brainscape)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert assert_keys.issubset(batch.keys())
        if "class_labels" in assert_keys:
            assert torch.equal(batch["class_labels"], torch.tensor([0]))


def test_brainscape_latent_and_image_with_conditioning(tmp_path):
    root, json_path = _make_json(tmp_path, with_image=True)
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            f"dataset.brainscape.json_path={json_path}",
            f"dataset.brainscape.dataset_root={root}",
            "dataset.brainscape.num_workers=0",
            "dataset.brainscape.batch_size=1",
            "dataset.brainscape.val_batch_size=1",
            "dataset.brainscape.test_batch_size=1",

            "dataset.brainscape.modalities=[t1w]",


            "dataset.brainscape.input_specs=[{key:image,group:preprocessed,modalities:[t1w]},{key:latent,group:latent,modalities:[t1w]}]",
        ]
        cfg = compose(config_name="data/minimal_brainscape", overrides=overrides)
        dm = instantiate(cfg.dataset.brainscape)
        tf = build_simple_tf()
        dm._tf["train"] = tf
        dm._tf["val"] = tf
        dm._tf["test"] = tf
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert set(batch.keys()) >= {"image", "latent", "class_labels"}
        assert batch["image"].shape[-3:] == batch["latent"].shape[-3:]


def test_brainscape_extra_only(tmp_path):
    root, json_path = _make_json(tmp_path)
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            f"dataset.brainscape.json_path={json_path}",
            f"dataset.brainscape.dataset_root={root}",
            "dataset.brainscape.num_workers=0",
            "dataset.brainscape.batch_size=1",
            "dataset.brainscape.val_batch_size=1",
            "dataset.brainscape.test_batch_size=1",


            "dataset.brainscape.input_specs=[{key:latent,group:latent,modalities:[t1w]}]",
            "dataset.brainscape.modalities=[t1w]",
            "dataset.train_tf._target_=brainmint.data.transforms.mri_vae.VAETransform",
            "dataset.train_tf.is_train=false",
            "dataset.train_tf.random_aug=false",
            "dataset.train_tf.image_keys=[]",
            "dataset.train_tf.passthrough_keys=[latent]",
            "dataset.train_tf.meta_keys=[modality]",

            "dataset.train_tf.brain_roi_size=[4,4,4]",
            "dataset.train_tf.patch_size=[4,4,4]",
            "+dataset.train_tf.val_patch_size=null",
            "dataset.val_tf=${dataset.train_tf}",
            "dataset.test_tf=${dataset.train_tf}",
        ]
        cfg = compose(config_name="data/minimal_brainscape", overrides=overrides)
        dm = instantiate(cfg.dataset.brainscape)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert {"latent"}.issubset(batch.keys())

