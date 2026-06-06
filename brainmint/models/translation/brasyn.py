from __future__ import annotations

"""BrainMint-facing BraSyn / BrainLesion MissingMRI translation wrappers."""

from typing import Any, Mapping

import torch

from brainmint.integrations.brasyn.missing_mri import BraSynMissingMRIGenerator


class BraSynMissingMRITranslator(BraSynMissingMRIGenerator):
    """Thin translation wrapper around the BrainLesion MissingMRI runner.

    The integration layer owns the upstream ``brats`` package, container patching,
    temporary NIfTI conversion, and algorithm-specific runtime behavior. This
    class only exposes the BrainMint translation-facing call shape.
    """

    def translate(self, *, batch: Mapping[str, Any], target_modality: str) -> torch.Tensor:
        return self(batch=batch, modality=target_modality)

    def forward(self, *, batch: Mapping[str, Any], modality: str) -> torch.Tensor:
        return self.translate(batch=batch, target_modality=modality)


class BraSyn2023Pix2PixTranslator(BraSynMissingMRITranslator):
    """BraSyn 2023 Pix2Pix MissingMRI baseline (BraTS23_1)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(algorithm="BraTS23_1", **kwargs)


class BraSyn2024HFGANRefinerTranslator(BraSynMissingMRITranslator):
    """BraSyn 2024 HF-GAN + 3D Refiner MissingMRI baseline (BraTS24_1)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(algorithm="BraTS24_1", **kwargs)


class BraSyn2024MMCLACAGANTranslator(BraSynMissingMRITranslator):
    """BraSyn 2024 MMCL / ACA-GAN MissingMRI baseline (BraTS24_3)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(algorithm="BraTS24_3", **kwargs)
