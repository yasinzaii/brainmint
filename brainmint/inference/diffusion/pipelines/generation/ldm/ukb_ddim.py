from __future__ import annotations

from brainmint.inference.diffusion.pipelines.base import LatentDiffusionGenerationPipeline


class UkbLdmGenerationPipeline(LatentDiffusionGenerationPipeline):
    """Thin alias for UKB-LDM generation.

    Keeping this class (even though it currently adds no behavior) is useful because:
      - Hydra configs can target this class by name,
      - future UKB-LDM-specific defaults/checks can be added without refactoring configs.
    """

    pass
