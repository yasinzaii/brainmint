from __future__ import annotations

"""BrainMint-facing ALDM VQ-GAN compression wrapper."""

from pathlib import Path
from typing import Any, Optional, Union

from torch import nn

from brainmint.integrations.aldm.vqgan import build_stage1_vqgan


class ALDMVQGAN(nn.Module):
    """Hydra-compatible wrapper around ALDM's upstream stage-1 VQ-GAN.

    The ALDM integration reads the upstream stage-1 config by default and
    patches only inference-specific values, so BrainMint configs do not duplicate
    the external architecture.
    """

    def __init__(
        self,
        *,
        ckpt_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.model = build_stage1_vqgan(
            checkpoint_path=ckpt_path,
            config_path=config_path,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self.model(*args, **kwargs)


__all__ = ["ALDMVQGAN"]
