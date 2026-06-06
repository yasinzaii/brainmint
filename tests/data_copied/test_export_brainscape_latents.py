import sys
import json
from pathlib import Path

import torch
import numpy as np
import nibabel as nib
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from brainmint.lightning.export_latents_module import ExportLatentsModule
from experiments.maisi.export_latents import _clone_json_with_latents


CONFIG_DIR = PROJECT_ROOT / "configs"


def _write_nifti(path: Path, shape=(32, 32, 32)) -> None:
    img = nib.Nifti1Image(np.random.rand(*shape).astype("float32"), affine=np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)


def test_export_brainscape_latent_module(tmp_path):
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
            {"dataset": "HCP", "subject": "v1", "preprocessed": {"t1ce": "v1_t1ce.nii.gz"}},
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
            # Paths Overides
            f"paths.brainscape_prep={src_root}",
            f"paths.brainscape_json={src_json}",
            f"paths.masi_brainscape_prep={out_root}",
            f"paths.masi_brainscape_json={out_json}",
            
            # BrainScape Overrides
            "dataset.brainscape.num_workers=0",
            "dataset.brainscape.subset_frac=null",
            
            # Trainer Overides
            "trainer.pl_trainer.accelerator=cpu",
            "trainer.pl_trainer.devices=1",
            
            # Model Overrides
            "model.autoencoder.autoencoder_maisi.num_channels=[8,16]",
            "model.autoencoder.autoencoder_maisi.num_res_blocks=[1,1]",
            "model.autoencoder.autoencoder_maisi.latent_channels=4",
            "model.autoencoder.autoencoder_maisi.norm_float16=false",
            "model.autoencoder.autoencoder_maisi.norm_num_groups=1",
            "model.autoencoder.autoencoder_maisi.attention_levels=[false,false]",
        ]
        
        cfg = compose(config_name="exp/maisi/brainscape_latents", overrides=overrides)
        print(OmegaConf.to_yaml(cfg, resolve=True))
        
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
        module._sum_sq.zero_()
        module._count.zero_()
        trainer.predict(module, dataloaders=dm.train_dataloader())
        trainer.predict(module, dataloaders=dm.val_dataloader())
        trainer.predict(module, dataloaders=dm.test_dataloader())

    _clone_json_with_latents(src_json, out_json, out_root)

    with out_json.open() as f:
        out = json.load(f)

    assert set(out.keys()) == {'test', 'train', 'val'}
    assert len(out["train"]) == 2
    assert len(out["val"]) == 1

    train_rec = out["train"][0]
    assert set(train_rec["latent"].keys()) == {"t1w", "t2w"}
    assert train_rec["latent"]["t1w"].endswith("s1_t1w_emb_mu.nii.gz")
    assert train_rec["recon"]["t2w"].endswith("s1_t2w_recon.nii.gz")

    lat_path = out_root / train_rec["dataset"] / "preprocessed" / train_rec["latent"]["t1w"]
    recon_path = out_root / train_rec["dataset"] / "preprocessed" / train_rec["recon"]["t1w"]
    pre_path = out_root / train_rec["dataset"] / "preprocessed" / train_rec["preprocessed"]["t1w"]
    assert nib.load(str(recon_path)).shape == tuple(cfg.dataset.brain_roi_size)  # Resized by Transform
    lat_shape = nib.load(str(lat_path)).shape[1:]
    # The transforms pads the input image (32 pix cube) - So latent is bigger than input 
    #assert all(ls <= ps for ls, ps in zip(lat_shape, nib.load(str(pre_path)).shape))