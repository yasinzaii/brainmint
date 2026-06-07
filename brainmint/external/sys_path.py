# brainmint/external/sys_path.py
from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

PathLike = str | Path


def resolve_path(path: str | Path) -> Path:
    # strict=False works even for paths that do not exist yet
    p = Path(path).expanduser()
    try:
        return p.resolve(strict=False)  # py3.9+
    except TypeError:
        return p.resolve()


@contextmanager
def repo_on_syspath(paths: Sequence[PathLike]) -> Iterator[None]:
    """Temporarily prepend paths to sys.path (scoped)."""
    normed = [str(resolve_path(p)) for p in paths]
    old = list(sys.path)
    try:
        for p in reversed(normed):
            if p not in sys.path:
                sys.path.insert(0, p)
        yield
    finally:
        sys.path[:] = old
