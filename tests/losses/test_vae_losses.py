# tests/test_vae_loss_manager.py
import importlib
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "configs"


m = importlib.import_module("brainmint.losses.vae_loss_manager")
print("Module OK, VaeLossManager:", getattr(m, "VaeLossManager", None))

@pytest.mark.parametrize("loss_cfg", ["/loss/vae_loss"])  
def test_vae_loss_manager_forward(loss_cfg):
    """
    Instantiate the VaeLossManager from Hydra config, run one generator
    and discriminator loss pass with dummy tensors.
    """
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        
        cfg = compose(config_name=loss_cfg)

        # 2. cfg.vae_loss is _partial_: true → returns a partial object
        loss_mgr_factory = instantiate(cfg.loss.vae_loss)

        # 3. Finish construction with runtime-only arguments
        loss_mgr = loss_mgr_factory(
            kl_weight=1.6e-7,
            adv_weight=0.1,
            perceptual_weight=0.3,
            device=torch.device("cpu"),
        )
        loss_mgr.set_mode(mode="train")
        # 4. Create dummy tensors
        batch, ch, d, h, w = 2, 1, 32, 32, 32
        zd, zh, zw = 4, 4, 4  # latent dims                     
        images  = torch.randn(batch, ch, d, h, w)
        recon   = torch.randn_like(images)           # random "reconstruction"
        z_mu    = torch.zeros(batch, ch, zd, zh, zw) # simple, stable values
        z_sigma = torch.zeros_like(z_mu)             # log-var = 0  → σ² = 1
        logits_fake = torch.randn(batch, 1)
        logits_real = torch.randn(batch, 1)

        # 5. Run generator & discriminator losses
        g_loss = loss_mgr.gen_loss(
            images=images,
            recon=recon,
            z_mu=z_mu,
            z_sigma=z_sigma,
            logits_fake=logits_fake,
            bs=batch,
        )
        d_loss = loss_mgr.disc_loss(
            logits_real=logits_real,
            logits_fake=logits_fake,
            bs=batch,
        )

        # 6. Assertions
        assert torch.isfinite(g_loss).all(), "Generator loss is NaN/Inf"
        assert torch.isfinite(d_loss).all(), "Discriminator loss is NaN/Inf"

        # Generator & discriminator called once each → consistency check passes
        loss_mgr.check_consistency()

        # Book-keep & make sure history is recorded
        loss_mgr.store_epoch(mode="train")
        assert len(loss_mgr.get_train_loss_rec()) == 1, "Epoch history not stored"
