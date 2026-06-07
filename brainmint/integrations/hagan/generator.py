"""HA-GAN upstream generator builders."""

from __future__ import annotations

from torch import nn

from brainmint.integrations.hagan.repo import hagan_repo_context


def build_hagan_generator(
    *,
    resolution: int = 256,
    mode: str = "eval",
    latent_dim: int = 1024,
    channel: int = 32,
    num_class: int = 0,
) -> nn.Module:
    """Build the upstream HA-GAN generator without loading checkpoints."""

    resolution = int(resolution)
    if resolution not in (128, 256):
        raise ValueError(f"Unsupported HA-GAN resolution {resolution}. Expected 128 or 256.")

    with hagan_repo_context():
        if resolution == 256:
            from models.Model_HA_GAN_256 import Generator
        else:
            from models.Model_HA_GAN_128 import Generator

    return Generator(
        mode=str(mode),
        latent_dim=int(latent_dim),
        channel=int(channel),
        num_class=int(num_class),
    )
