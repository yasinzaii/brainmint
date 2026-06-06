import sys
from pathlib import Path

import torch
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR   = PROJECT_ROOT / "configs"

@pytest.mark.parametrize("dataset_cfg", ["dataset/brainscape_test"])
def test_maisi_autoencoder_single_forward(dataset_cfg):
    """
    Instantiate the MAISI autoencoder and run a single forward pass on one
    BrainScape sample. 
    """
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        # compose dataset and autoencoder configs
        data_cfg  = compose(config_name=dataset_cfg)

        model_cfg = compose(
            config_name="model/autoencoder/maisi",
            overrides=["model.autoencoder.autoencoder_maisi.norm_float16=false"],
        )

        # build DataModule and take one image sample
        dm = instantiate(data_cfg.dataset.brainscape)
        dm.setup()
        sample = next(iter(dm.train_dataloader()))["image"][:1]  # shape (1, C, D, H, W)
        sample = sample.float() # Convert to FP32 if FP16

        # instantiate the autoencoder
        ae = instantiate(model_cfg.model.autoencoder.autoencoder_maisi)

        # run a forward pass
        recon = ae(sample)
        assert recon[0].shape == sample.shape, "Output shape must match input shape"
        assert torch.isfinite(recon[0]).all(), "Output contains NaNs or Infs"
