"""Batch-shape helpers for inference runners and pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def infer_batch_size(batch: Mapping[str, Any], *, default: int = 1) -> int:
    """Infer requested batch size from an inference batch.

    The explicit ``batch_size`` key wins. Otherwise, the first tensor with a
    batch dimension is used. This supports minimal generation requests such as
    ``{"batch_size": 4}`` without requiring a dataset-backed batch.
    """

    if "batch_size" in batch:
        value = batch["batch_size"]
        if torch.is_tensor(value):
            if value.ndim == 0:
                return int(value.item())
            if value.shape:
                return int(value.shape[0])
            return int(default)
        return int(value)

    for value in batch.values():
        if torch.is_tensor(value) and value.shape:
            return int(value.shape[0])

    return int(default)
