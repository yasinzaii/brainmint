from __future__ import annotations

"""Installed-MONAI MAISI autoencoder builders."""

from typing import Any

from torch import nn


def build_maisi_autoencoder(**kwargs: Any) -> nn.Module:
    """Build MONAI's MAISI AutoencoderKlMaisi from Hydra kwargs."""

    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

    return AutoencoderKlMaisi(**kwargs)
