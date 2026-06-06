import sys
import json
import pytest
from pathlib import Path

import torch
import numpy as np
import nibabel as nib

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "configs"

def _make_unet(cfg_name: str = "model/diffusion/maisi", num_classes: int | None = 4):
    """Instantiate MAISI diffusion model via Hydra configuration."""

    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=cfg_name,
            overrides=[
                "model.diffusion.diffusion_maisi.use_flash_attention=false",
                f"model.diffusion.diffusion_maisi.num_channels={[8, 16, 16, 32]}",
                f"model.diffusion.diffusion_maisi.norm_num_groups={4}",
                f"model.diffusion.diffusion_maisi.num_head_channels={[0, 0, 4, 4]}",
                f"model.diffusion.diffusion_maisi.num_class_embeds={num_classes if num_classes is not None else 'null'}",
            ],
        )
        return instantiate(cfg.model.diffusion.diffusion_maisi), cfg


def _write_latent(path: Path, spatial: tuple[int, int, int], channels: int = 4) -> None:
    """Write a random latent tensor to ``path`` as a NIfTI image.

    The NIfTI shape is ``(X, Y, Z, C)`` so that MONAI can convert it to channel-first format.
    """
    arr = np.random.rand(*spatial, channels).astype("float32")
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)


def _generate_dataset(
    root: Path,
    json_path: Path,
    data: dict,
    spatial_map: dict[str, tuple[int, int, int]],
) -> None:
    """Write ``data`` JSON and populate ``root`` with random NIfTI files.

    ``spatial_map`` maps tensor keys (e.g. ``"latent"``) to spatial sizes.  A
    default of 4 channels is used for ``latent`` tensors and 1 otherwise.
    """

    with json_path.open("w") as f:
        json.dump(data, f)

    for split in data.values():
        for rec in split:
            for key, rels in rec.items():
                if key in spatial_map and isinstance(rels, dict):
                    shape = spatial_map[key]
                    channels = 4 if key == "latent" else 1
                    for rel in rels.values():
                        _write_latent(root / rec["dataset"] / "preprocessed" / rel, shape, channels)


@pytest.mark.parametrize("num_classes", [None, 4])
@pytest.mark.parametrize("spatial", [(8, 8, 8), (16, 16, 16), (48, 56, 40)] ) 
def test_unet_classcond_forward_and_grads(num_classes, spatial):
    """Forward pass works for varying input sizes and optional class labels."""

    unet, cfg = _make_unet(num_classes=num_classes)

    B, C = 2, cfg.model.shared.latent_channels
    z, y, x = spatial
    x_in = torch.randn(B, C, z, y, x)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)

    kwargs = {}
    if num_classes is not None:
        kwargs["class_labels"] = torch.tensor([0, 3], dtype=torch.long)  # 0-based

    out = unet(x=x_in, timesteps=t, **kwargs)
    assert out.shape == x_in.shape
    assert torch.isfinite(out).all()

    out.sum().backward()
    w = unet.conv_in.conv.weight
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_unet_requires_labels_when_classcond_enabled():
    """If ``num_class_embeds`` > 0, omitting ``class_labels`` must raise."""

    unet, cfg = _make_unet(num_classes=4)
    B, C = 2, cfg.model.shared.latent_channels
    x_in = torch.randn(B, C, 8, 8, 8)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)

    with pytest.raises(ValueError):
        _ = unet(x=x_in, timesteps=t)  # missing class_labels


def test_unet_out_of_range_label_raises():
    """Class labels must lie in [0, ``num_class_embeds``-1]."""

    unet, cfg = _make_unet(num_classes=4)
    B, C = 2, cfg.model.shared.latent_channels
    x_in = torch.randn(B, C, 8, 8, 8)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)
    bad = torch.tensor([0, 4], dtype=torch.long)  # 4 is out-of-range

    with pytest.raises((RuntimeError, IndexError)):
        _ = unet(x=x_in, timesteps=t, class_labels=bad)


def test_unet_wrong_label_dtype_raises():
    """``nn.Embedding`` requires Long dtype; wrong dtype should error."""

    unet, cfg = _make_unet(num_classes=4)
    B, C = 2, cfg.model.shared.latent_channels
    x_in = torch.randn(B, C, 8, 8, 8)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)
    wrong_dtype = torch.tensor([0.0, 1.0], dtype=torch.float32)

    with pytest.raises((RuntimeError, TypeError)):
        _ = unet(x=x_in, timesteps=t, class_labels=wrong_dtype)


latent_specs = "[{key:latent,group:latent,modalities:[t1w,t2w]}]"

@pytest.mark.parametrize("input_specs", [latent_specs]) # We Train Diffusion Module only with latent images
def test_unet_with_brainscape_datamodule(tmp_path, input_specs ):
    """Integration test with ``BrainScapeDataModule`` and modality labels."""

    ds_root = tmp_path / "ds"
    json_path = tmp_path / "data.json"

    data = {
        "train": [
            {
                "dataset": "DS1",
                "subject": "sub-train1",
                "latent": {"t1w": "s1_t1w.nii.gz", "t2w": "s1_t2w.nii.gz"},
            },
            {
                "dataset": "DS2",
                "subject": "sub-train2",
                "latent": {"t1w": "s2_t1w.nii.gz", "t2w": "s2_t2w.nii.gz"},
            },
        ],
        "val": [
            {
                "dataset": "DS1",
                "subject": "sub-val",
                "latent": {"t1w": "v_t1w.nii.gz", "t2w": "v_t2w.nii.gz"},
            }
        ],
        "test": [
            {
                "dataset": "DS2",
                "subject": "sub-test",
                "latent": {"t1w": "t_t1w.nii.gz", "t2w": "t_t2w.nii.gz"},
            }
        ],
    }

    _generate_dataset(ds_root, json_path, data, {"latent": (8, 8, 8)})

    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        img_keys = "[image]" if "image" in input_specs else "[]"
        label_keys = "[label]" if "label" in input_specs else "[]"
        latent_keys = "[latent]" if "latent" in input_specs else "[]"
        
        overrides = [
            # Paths Overrides
            f"paths.brainscape_prep={ds_root}",
            f"paths.brainscape_json={json_path}",
            
            # BrainScape Dataset Overrides
            "dataset.brainscape.num_workers=0",
            "dataset.brainscape.subset_frac=null",
            f"dataset.brainscape.input_specs={input_specs}",
            "dataset.brainscape.batch_size=2",
            "dataset.brainscape.val_batch_size=2",
            "dataset.brainscape.test_batch_size=2",

            # Latent are passed-through by transform ROI size for ('image')
            #f"dataset.brain_roi_size={[8,8,8]}",

            # Transforms Override
            f"dataset.train_tf.image_keys={img_keys}",
            f"dataset.train_tf.label_keys={label_keys}",
            f"dataset.train_tf.passthrough_keys={latent_keys}",
            "dataset.train_tf.extra_xforms_end=[]",
            f"dataset.val_tf.image_keys={img_keys}",
            f"dataset.val_tf.label_keys={label_keys}",
            f"dataset.val_tf.passthrough_keys={latent_keys}",
            "dataset.val_tf.extra_xforms_end=[]",
            "dataset.synthetic_mask.synthetic_mask.mask_pool=[]",
        ]

        cfg = compose(config_name="dataset/brainscape", overrides=overrides)

        dm = instantiate(cfg.dataset.brainscape)
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch["latent"].shape == (2, 4, 8, 8, 8)
        mods = set(batch["modality_map"].tolist())
        for b in loader:
            mods.update(b["modality_map"].tolist())
        assert mods == {0, 1}

        # Instantiate model and run a forward pass with class conditioning
        cfg_model = compose(
            config_name="model/diffusion/maisi",
            overrides=[
                "model.diffusion.diffusion_maisi.use_flash_attention=false",
                "model.diffusion.diffusion_maisi.num_class_embeds=4",
            ],
        )
        unet = instantiate(cfg_model.model.diffusion.diffusion_maisi)

        t = torch.randint(0, 1000, (2,), dtype=torch.long)
        out = unet(x=batch["latent"], timesteps=t, class_labels=batch["modality_map"])

        assert out.shape == batch["latent"].shape
        assert torch.isfinite(out).all()



@pytest.mark.parametrize("input_specs", [latent_specs])
def test_real_brainscape_input_spec_combinations(input_specs):
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        img_keys = "[image]" if "image" in input_specs else "[]"
        label_keys = "[label]" if "label" in input_specs else "[]"
        latent_keys = "[latent]" if "latent" in input_specs else "[]"
        overrides = [
            
            # Path Overides
            "paths.brainscape_prep=${paths.masi_brainscape_prep_test}",
            "paths.brainscape_json=${paths.masi_brainscape_json_test}",
            
            # BrainScape Test Dataset Overrides
            #"dataset.brain_roi_size=[64,64,64]", # Not applied - Latents are passed-through - Transform
            "dataset.brainscape.num_workers=0",
            f"dataset.brainscape.input_specs={input_specs}",
            "dataset.brainscape.subset_frac=null",
            
            # Tranform Overrides
            f"dataset.train_tf.image_keys={img_keys}",
            f"dataset.train_tf.label_keys={label_keys}",
            f"dataset.train_tf.passthrough_keys={latent_keys}",
            "dataset.train_tf.extra_xforms_end=[]",
            "dataset.train_tf.meta_keys=[modality]",
            "dataset.conditioning.modality_conditioning.key_name=class_labels",
            "dataset.synthetic_mask.synthetic_mask.mask_pool=[]",
        ]
        cfg = compose(config_name="dataset/brainscape_test", overrides=overrides)

    prep = Path(cfg.paths.brainscape_prep)
    jsn = Path(cfg.paths.brainscape_json)
    if not prep.exists() or not jsn.exists():
        pytest.skip("BrainScape test dataset not available")

    dm = instantiate(cfg.dataset.brainscape)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    unet, _ = _make_unet(num_classes=4)
    t = torch.randint(0, 1000, (batch["latent"].shape[0],), dtype=torch.long)
    kwargs = {"class_labels": batch["class_labels"]} if "class_labels" in batch else {}
    out = unet(x=batch["latent"], timesteps=t, **kwargs)
    assert out.shape == batch["latent"].shape
    assert torch.isfinite(out).all()

