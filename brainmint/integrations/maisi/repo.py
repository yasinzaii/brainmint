from __future__ import annotations

"""MAISI / NV-Generate-MR external repository helpers."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

from brainmint.external.registry import get_repo_spec
from brainmint.external.repo_manager import import_external_repo

PathLike = Union[str, Path]

MAISI_REPO_NAME = "maisi"


@contextmanager
def maisi_repo_context(
    *,
    external_repo_root: Optional[PathLike] = None,
    allow_network: bool = True,
) -> Iterator[Path]:
    """Temporarily expose the canonical NV-Generate-MR repo on ``sys.path``."""

    spec = get_repo_spec(MAISI_REPO_NAME)
    with import_external_repo(
        spec,
        external_repo_root=external_repo_root,
        allow_network=allow_network,
    ) as resolved_repo_root:
        yield resolved_repo_root
