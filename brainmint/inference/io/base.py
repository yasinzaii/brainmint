from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np


PathLike = Union[str, Path]


class ReaderBase(ABC):
    """Base class for readers used by inference scripts."""

    @abstractmethod
    def read(self, path: PathLike, *, meta: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        raise NotImplementedError


class WriterBase(ABC):
    """Base class for writers used by inference scripts.

    Writers are intentionally kept outside Lightning and pipelines so that
    inference stays pure and I/O policy can change independently.
    """

    @abstractmethod
    def write(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def flush(self) -> None:  # pragma: no cover - optional hook
        return None
