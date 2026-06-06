from typing import Any, Dict, Mapping, Optional

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.diffusion.conditioning.base import ConditioningBuilderBase


class ClassLabelsConditioning(ConditioningBuilderBase):
    """Extract class labels from a batch.

    Output:
      - ``out_key``: (B,) long

    Assumptions (intentional):
      - ``batch[in_key]`` is a tensor of shape ``(B,)`` or ``(B,1)``.
      - ``B`` matches the current latent batch size.
    """

    def __init__(
        self,
        *,
        in_key: str = "modality_map",
        out_key: str = "class_labels",
        num_classes: Optional[int] = None,
        mapping: Optional[Mapping[str, int]] = None,
        fail_on_unknown: bool = True,
    ) -> None:
        super().__init__()
        self.in_key = str(in_key)
        self.out_key = str(out_key)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.mapping = None if mapping is None else {str(k).lower(): int(v) for k, v in mapping.items()}
        self.fail_on_unknown = bool(fail_on_unknown)

    def build(
        self,
        batch: Mapping[str, Any],
        *,
        latent_ref: torch.Tensor,
        ctx: InferenceContext,  # noqa: ARG002
    ) -> Dict[str, torch.Tensor]:
        if self.in_key not in batch:
            raise KeyError(f"Batch missing key '{self.in_key}'")

        x = batch[self.in_key]
        if torch.is_tensor(x):
            if x.ndim == 2 and x.shape[1] == 1:
                x = x[:, 0]
            if x.ndim != 1:
                raise ValueError(f"Batch[{self.in_key!r}] expected shape (B,) or (B,1), got {tuple(x.shape)}")
            labels = x.to(device=ctx.device, dtype=torch.long)
        else:
            if self.mapping is None:
                raise TypeError(
                    f"Batch[{self.in_key!r}] must be torch.Tensor unless mapping is provided; got {type(x)}"
                )
            if isinstance(x, (list, tuple)):
                items = list(x)
            else:
                items = [x]
            mapped = []
            for item in items:
                key = str(item).strip().lower()
                if key in self.mapping:
                    mapped.append(self.mapping[key])
                elif self.fail_on_unknown:
                    raise KeyError(f"Unknown class label '{item}'. Known: {sorted(self.mapping.keys())}")
                else:
                    mapped.append(0)
            labels = torch.tensor(mapped, device=ctx.device, dtype=torch.long)

        b = int(latent_ref.shape[0])
        if int(labels.shape[0]) != b:
            raise ValueError(f"{self.in_key} has B={int(labels.shape[0])} but latent_ref has B={b}")

        if self.num_classes is not None:
            mn = int(labels.min().item())
            mx = int(labels.max().item())
            if mn < 0 or mx >= self.num_classes:
                raise ValueError(f"class_labels out of range: min={mn} max={mx} for num_classes={self.num_classes}")

        return {self.out_key: labels}
