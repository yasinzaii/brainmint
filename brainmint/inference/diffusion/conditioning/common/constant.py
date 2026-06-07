from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence as ABCSequence
from typing import Any

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.diffusion.conditioning.base import ConditioningBuilderBase

try:
    # Hydra/OmegaConf is a core dependency in this repo, but keep this optional to avoid hard import errors
    # in minimal environments.
    from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    DictConfig = ()  # type: ignore
    ListConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore


Scalar = int | float | bool
Value = Scalar | ABCSequence | torch.Tensor


class ConstantConditioning(ConditioningBuilderBase):
    """Produce constant tensor conditioning values.

    This is useful for models like MAISI where "structured conditioning" is required even when you
    are not using per-sample metadata.

    Notes:
      - Values may come from Hydra as OmegaConf ListConfig. We normalize those to plain lists.
      - Scalars are broadcast to (B,)
      - 1D vectors are broadcast to (B, D)
      - 2D tensors with leading dim 1 are broadcast to (B, ...)
    """

    def __init__(
        self,
        *,
        constants: Mapping[str, Any],
        allow_override_from_batch: bool = False,
        dtypes: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.constants = dict(constants)
        self.allow_override_from_batch = bool(allow_override_from_batch)
        self._dtype_overrides = dict(dtypes) if dtypes is not None else {}

    def _dtype_from_name(self, key: str) -> torch.dtype:
        """Heuristic dtype selection.

        Users can override per-key via `dtypes` in Hydra config:
          dtypes:
            class_labels: long
        """
        if key in self._dtype_overrides:
            v = str(self._dtype_overrides[key]).lower()
            if v in {"long", "int64"}:
                return torch.long
            if v in {"int", "int32"}:
                return torch.int32
            if v in {"float", "float32"}:
                return torch.float32
            if v in {"float16", "half"}:
                return torch.float16
            if v in {"bfloat16"}:
                return torch.bfloat16
            if v in {"bool"}:
                return torch.bool
            raise ValueError(f"Unknown dtype override '{self._dtype_overrides[key]}' for key '{key}'")

        # Default heuristic
        lk = key.lower()
        if lk in {"class_labels", "labels"} or lk.endswith("_labels") or lk.endswith("_label"):
            return torch.long
        return torch.float32

    @staticmethod
    def _maybe_omegaconf_to_container(v: Any) -> Any:
        if OmegaConf is None:
            return v
        if isinstance(v, (ListConfig, DictConfig)):
            return OmegaConf.to_container(v, resolve=True)
        return v

    def build(
        self,
        batch: Mapping[str, Any],
        *,
        latent_ref: torch.Tensor,
        ctx: InferenceContext,
    ) -> dict[str, torch.Tensor]:
        b = int(latent_ref.shape[0])
        out: dict[str, torch.Tensor] = {}

        for k, v0 in self.constants.items():
            v = v0
            # Optional: allow batch to override constants by key (useful for experiments)
            if self.allow_override_from_batch and (k in batch):
                v = batch[k]

            v = self._maybe_omegaconf_to_container(v)
            dt = self._dtype_from_name(k)

            # Tensor case
            if torch.is_tensor(v):
                t = v.to(device=ctx.device)
                # Broadcast as needed
                if t.dim() == 0:
                    t = t.view(1).expand(b)
                elif t.dim() == 1:
                    if t.shape[0] != b:
                        t = t.unsqueeze(0).expand(b, -1)
                else:
                    if t.shape[0] == 1 and b != 1:
                        t = t.expand(b, *t.shape[1:])
                if int(t.shape[0]) != b:
                    raise ValueError(f"ConstantConditioning key '{k}' has B={int(t.shape[0])} but expected B={b}")
                out[k] = t.to(dtype=dt)
                continue

            # Scalar -> (B,)
            if isinstance(v, (int, float, bool)):
                out[k] = torch.tensor([v] * b, device=ctx.device, dtype=dt)
                continue

            # Sequence (including OmegaConf ListConfig after to_container)
            # Exclude strings/bytes: those are sequences too but not numeric conditioning.
            if isinstance(v, ABCSequence) and not isinstance(v, (str, bytes, bytearray)):
                if len(v) == 0:
                    raise ValueError(f"ConstantConditioning key '{k}' has empty sequence value")
                # Convert to a plain list first (ListConfig -> list, tuple -> list, etc.)
                vv = list(v)
                t = torch.tensor(vv, device=ctx.device, dtype=dt)
                # Broadcast 1D vector to (B, D)
                if t.dim() == 0:
                    t = t.view(1).expand(b)
                elif t.dim() == 1:
                    t = t.unsqueeze(0).expand(b, -1)
                else:
                    # If it's already 2D+ and leading dim 1, broadcast
                    if t.shape[0] == 1 and b != 1:
                        t = t.expand(b, *t.shape[1:])
                if int(t.shape[0]) != b:
                    raise ValueError(f"ConstantConditioning key '{k}' has B={int(t.shape[0])} but expected B={b}")
                out[k] = t
                continue

            raise TypeError(
                f"ConstantConditioning key '{k}' has unsupported value type: {type(v)} "
                f"(original={type(v0)})"
            )

        return out
