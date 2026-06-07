"""cWDM external repository helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from brainmint.external.registry import get_repo_spec
from brainmint.external.repo_manager import import_external_repo

PathLike = str | Path

CWDM_REPO_NAME = "cwdm"


@contextmanager
def cwdm_repo_context(
    *,
    external_repo_root: PathLike | None = None,
    allow_network: bool = True,
) -> Iterator[Path]:
    """Temporarily expose the canonical cWDM repo on ``sys.path``."""

    spec = get_repo_spec(CWDM_REPO_NAME)
    with import_external_repo(
        spec,
        external_repo_root=external_repo_root,
        allow_network=allow_network,
    ) as resolved_repo_root:
        yield resolved_repo_root
