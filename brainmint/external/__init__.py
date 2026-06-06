"""External repository registry and import-path management.

This package owns BrainMint package-safe external repo root policy. External
model repositories are resolved from an explicit process override, the
BRAINMINT_EXTERNAL_ROOT environment variable, or a platform-native user cache.
"""

from brainmint.external.repo_manager import (
    BRAINMINT_EXTERNAL_ROOT_ENV,
    ExternalRepoManager,
    get_external_repo_root,
    set_external_repo_root,
)

__all__ = [
    "BRAINMINT_EXTERNAL_ROOT_ENV",
    "ExternalRepoManager",
    "get_external_repo_root",
    "set_external_repo_root",
]
