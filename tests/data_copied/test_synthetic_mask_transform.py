import sys
import json
from pathlib import Path

import torch
import pytest
import numpy as np
import nibabel as nib
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "configs"

from brainmint.data.transforms.synthetic_mask import SyntheticMaskTransform
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd



@pytest.mark.parametrize("num_workers", [0])
def test_synthetic_mask_transform_brainscape_datamodule(tmp_path, num_workers):
    ds_root = tmp_path / "data"
    ds_root.mkdir()
    #json_path = _generate_dataset(ds_root)

    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        ds_overrides = [
            #f"paths.brainscape_prep={ds_root}",
            #f"paths.brainscape_json={json_path}",
            "dataset.brainscape.num_workers=0",
            "dataset.brainscape.subset_frac=null",
            "dataset.brainscape.modalities=[T1w, t2W]",
            "dataset.brainscape.batch_size=1",
            "dataset.brainscape.val_batch_size=1",
            "dataset.brainscape.test_batch_size=1",
            "dataset.train_tf.extra_xforms_end=[${dataset.masks.synthetic_mask}]",
            "dataset.val_tf.extra_xforms_end=[${dataset.masks.synthetic_mask}]"
        ]

        cfg = compose(config_name="exp/maisi/train_controlnet", overrides=ds_overrides)
        print(OmegaConf.to_yaml(cfg, resolve=True))
        dm = instantiate(cfg.dataset.brainscape)
        dm.setup()

        loader = dm.train_dataloader()

        # JUST FOR BRATS - [ALREADY HAVE SEG!!!]
        counter_others = 0
        counter_brats = 0
        for idx, batch in enumerate(loader):
            sample = {k: (v[0] if isinstance(v, torch.Tensor) else v[0]) for k, v in batch.items()}
            dataset_name = sample["dataset"]
            if dataset_name != 'BRATS':
                if counter_others > 2:
                    continue
                else:
                    counter_others +=1
            else:
                counter_brats += 1
                
            case_dir = tmp_path / f"{idx}"
            case_dir.mkdir(parents=True, exist_ok=True)
            
            mask_path       = case_dir / f"{dataset_name}_{idx}_mask_dilated.nii.gz"
            img_masked_path = case_dir / f"{dataset_name}_{idx}_image_masked.nii.gz"
            img_orig_path   = case_dir / f"{dataset_name}_{idx}_image_original.nii.gz"
            onehot_path     = case_dir / f"{dataset_name}_{idx}_mask_onehot.nii.gz"
            color_mask_path = case_dir / f"{dataset_name}_{idx}_mask_color.nii.gz"   # single-channel label map (0,1,2,3)
            overlay_path    = case_dir / f"{dataset_name}_{idx}_overlay.nii.gz"      # single-channel overlay
            
            mask_dilated_dhw = sample["mask_dilated"][0].cpu().numpy().astype(np.uint8)  # (D,H,W)
            nib.save(nib.Nifti1Image(mask_dilated_dhw, np.eye(4)), mask_path)

            img_cdhw = sample["image"].cpu().numpy().astype(np.float32)  # (C,D,H,W)
            if img_cdhw.shape[0] == 1:
                img_orig_dhw = img_cdhw[0]                               # (D,H,W)
            else:
                img_orig_dhw = np.moveaxis(img_cdhw, 0, -1)              # (D,H,W,C)
            nib.save(nib.Nifti1Image(img_orig_dhw, np.eye(4)), img_orig_path)
    
            
            # Masked Image
            img_masked_cdhw = sample["image_masked"].cpu().numpy().astype(np.float32)  # (C,D,H,W)
            img_masked_dhwc = np.moveaxis(img_masked_cdhw, 0, -1)                      # (D,H,W,C)
            img_masked_save = img_masked_dhwc[..., 0] if img_masked_dhwc.shape[-1] == 1 else img_masked_dhwc
            nib.save(nib.Nifti1Image(img_masked_save, np.eye(4)), img_masked_path)


            mask_oh_cdhw = sample["mask_one_hot"].cpu().numpy().astype(np.float32)     # (4,D,H,W) [H, NCR/NET, ED, ET]
            mask_oh_dhwc = np.moveaxis(mask_oh_cdhw, 0, -1)                            # (D,H,W,4)
            nib.save(nib.Nifti1Image(mask_oh_dhwc.astype(np.uint8), np.eye(4)), onehot_path)

            # Single-channel label map (0/1/2/3)
            label_dhw = np.argmax(mask_oh_dhwc, axis=-1).astype(np.uint8)              # (D,H,W)
            nib.save(nib.Nifti1Image(label_dhw, np.eye(4)), color_mask_path)

            # Overlay (mask color on original image)
            base = img_cdhw[0] if img_cdhw.shape[0] >= 1 else img_orig_dhw             # (D,H,W)
            # normalize MRI to [0,1]
            base_min, base_max = float(base.min()), float(base.max())
            base_norm = (base - base_min) / (base_max - base_min + 1e-8)

            mri_gain = 1.5  
            base_boost = (base_norm * mri_gain).astype(np.float32) 

            overlay_scalar = base_boost.astype(np.float32)
            tumour_mask = label_dhw > 0
            overlay_scalar[tumour_mask] = label_dhw[tumour_mask].astype(np.float32)    # 1,2,3 in tumour

            nib.save(nib.Nifti1Image(overlay_scalar, np.eye(4)), overlay_path)
                
            # sanity checks
            assert mask_path.is_file()
            assert img_masked_path.is_file()
            assert img_orig_path.is_file()
            assert onehot_path.is_file()
            assert color_mask_path.is_file()
            assert overlay_path.is_file()
            assert set(np.unique(label_dhw).tolist()) <= {0, 1, 2, 3}
            
            if dataset_name != 'BRATS':
                assert bool(sample["mask_is_synthetic"]) is True
            else:
                assert bool(sample["mask_is_synthetic"]) is False

            
            if counter_brats > 2 and counter_others > 2:
                break

            


def test_synthetic_mask_transform_load_existing_mask(tmp_path):
    dim = (32, 32, 32)                 # (X, Y, Z)
    img_shape = (1, *dim)              # (C=1, X, Y, Z) -> (1,32,32,32)
    out_masks_shape = (4, *dim)        # (Classes=4, X, Y, Z)
    img = np.zeros(img_shape, dtype=np.float32)
    seg = np.zeros(img_shape, dtype=np.int16)
    seg[0, 4:14, 4:14, 4:14] = 2
    seg[0, 5:10, 5:10, 5:10] = 1
    mask_dir = tmp_path / "BRATS" / "preprocessed"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.moveaxis(seg, 0, -1).astype(np.uint8), affine=np.eye(4)), mask_path)
    json_path = tmp_path / "data.json"
    with json_path.open("w") as f:
        json.dump({
            "train": [
                {
                    "dataset": "BRATS",
                    "preprocessed": {"seg": mask_path.name}
                }
            ]
        }, f, indent=2)
        
    mask_loader = Compose([
        LoadImaged(keys=["seg"]),
        EnsureChannelFirstd(keys=["seg"]),
    ])

    
    transform = SyntheticMaskTransform(
        image_key="image",
        mask_key="seg",
        dataset_json=json_path,
        mask_loader_tf=mask_loader,
        dataset_root=tmp_path,
    )
    out = transform({"image": torch.tensor(img), "seg": torch.tensor(seg)[0,:]})
    keys = ['mask_one_hot', 'mask_dilated', 'image_masked']
    for k in keys:
        tar_img = out[k].numpy().astype(np.float32)
        tar_img_arr = np.moveaxis(tar_img, 0, -1).astype(np.uint8)   # (X,Y,Z,C)
        nib.save(nib.Nifti1Image(tar_img_arr, affine=np.eye(4)), tmp_path / f"{k}.nii.gz")

    assert tuple(out['mask_one_hot'].shape) == out_masks_shape
    assert tuple(out['mask_dilated'].shape) == img_shape
    assert tuple(out['image_masked'].shape) == img_shape
    assert out["mask_is_synthetic"] is False

    out = transform({"image": torch.tensor(img)})
    assert tuple(out['mask_one_hot'].shape) == out_masks_shape
    assert tuple(out['mask_dilated'].shape) == img_shape
    assert tuple(out['image_masked'].shape) == img_shape
    assert out["mask_is_synthetic"] is True

