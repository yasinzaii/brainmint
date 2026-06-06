from __future__ import annotations

"""Compatibility exports for external generation pipelines.

Prefer importing from the family-specific modules in this package. This module
remains so existing Hydra configs that target ``...pipelines.external`` keep
working while the package structure is cleaned up.
"""

from brainmint.inference.generation.pipelines.hagan import HaGANGenerationPipeline
from brainmint.inference.generation.pipelines.maisi import MAISI3DGenerationPipeline
from brainmint.inference.generation.pipelines.med_ddpm import MedDDPMGenerationPipeline
from brainmint.inference.generation.pipelines.wdm3d import WDM3DGenerationPipeline
