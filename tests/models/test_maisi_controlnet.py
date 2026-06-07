import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "configs"


def _make_controlnet(num_classes: int = 4, cond_in: int = 5):
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            "model.shared.latent_channels=4",
            "model.controlnet.controlnet_maisi.num_channels=[8,16]",
            "model.controlnet.controlnet_maisi.attention_levels=[false,false]",
            "model.controlnet.controlnet_maisi.num_head_channels=[4, 4]",
            "model.controlnet.controlnet_maisi.norm_num_groups=4",
            f"model.controlnet.controlnet_maisi.conditioning_embedding_in_channels={cond_in}",
            "model.controlnet.controlnet_maisi.conditioning_embedding_num_channels=[4]",
            f"model.controlnet.controlnet_maisi.num_class_embeds={num_classes}",
        ]
        cfg = compose(config_name="model/controlnet/maisi", overrides=overrides)
        return instantiate(cfg.model.controlnet.controlnet_maisi), cfg


def _make_diffusion_unet(num_classes: int = 4):
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        overrides = [
            "model.shared.latent_channels=4",
            "model.diffusion.diffusion_maisi.use_flash_attention=false",
            "model.diffusion.diffusion_maisi.num_channels=[8,16]",
            "model.diffusion.diffusion_maisi.attention_levels=[false, false]",
            "model.diffusion.diffusion_maisi.num_head_channels=[4,4]",
            "model.diffusion.diffusion_maisi.norm_num_groups=4",
            f"model.diffusion.diffusion_maisi.num_class_embeds={num_classes}",
        ]
        cfg = compose(config_name="model/diffusion/maisi", overrides=overrides)
        return instantiate(cfg.model.diffusion.diffusion_maisi), cfg


def test_controlnet_forward_and_grads():
    net, cfg = _make_controlnet(num_classes=2, cond_in=5)
    B = 2
    D = 16
    x = torch.randn(B, cfg.model.shared.latent_channels, D, D, D)
    cond = torch.randn(B, 5, D, D, D)
    t = torch.randint(0, 10, (B,), dtype=torch.long)
    labels = torch.tensor([0, 1], dtype=torch.long)
    downs, mid = net(x=x, timesteps=t, controlnet_cond=cond, class_labels=labels)
    assert isinstance(downs, list)
    total = sum(d.float().sum() for d in downs if torch.is_tensor(d))
    if mid is not None:
        total = total + mid.float().sum()
    total.backward()
    weights = next(net.parameters())
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


def test_controlnet_requires_labels_when_classcond_enabled():
    net, cfg = _make_controlnet(num_classes=2)
    x = torch.randn(1, cfg.model.shared.latent_channels, 8, 8, 8)
    cond = torch.randn(1, 5, 8, 8, 8)
    t = torch.randint(0, 10, (1,), dtype=torch.long)
    with pytest.raises(ValueError):
        net(x=x, timesteps=t, controlnet_cond=cond)


def test_controlnet_diffusion_integration():
    ctrl, ctrl_cfg = _make_controlnet(num_classes=2, cond_in=5)
    unet, _ = _make_diffusion_unet(num_classes=2)

    B = 2
    D = 8
    x = torch.randn(B, ctrl_cfg.model.shared.latent_channels, D, D, D)
    mask = torch.zeros(B, 4, D, D, D)
    mask[:, 0] = 1.0
    masked_img = torch.randn(B, 1, D, D, D)
    cond = torch.cat([mask, masked_img], dim=1)
    t = torch.randint(0, 10, (B,), dtype=torch.long)
    labels = torch.tensor([0, 1], dtype=torch.long)

    downs, mid = ctrl(x=x, timesteps=t, controlnet_cond=cond, class_labels=labels)
    assert isinstance(downs, list)
    assert all(tensor.shape[0] == B for tensor in downs if torch.is_tensor(tensor))

    downs_for_unet = [d.clone() if torch.is_tensor(d) else d for d in downs]
    mid_for_unet = mid.clone() if torch.is_tensor(mid) else mid
    if torch.is_tensor(mid_for_unet):
        assert mid_for_unet.shape[0] == B

    x_for_unet = x.clone()

    out = unet(
        x=x_for_unet,
        timesteps=t,
        down_block_additional_residuals=downs_for_unet,
        mid_block_additional_residual=mid_for_unet,
        class_labels=labels,
    )
    assert out.shape == x.shape
