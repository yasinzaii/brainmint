from __future__ import annotations

"""ALDM external repository layout helpers.

This module owns ALDM-specific repository discovery and import contexts. It does
not build models or load checkpoints; those responsibilities live in
``brainmint.integrations.aldm.vqgan`` and ``brainmint.integrations.aldm.ldm``.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

from brainmint.external.registry import get_repo_spec
from brainmint.external.repo_manager import ExternalRepoManager
from brainmint.external.sys_path import repo_on_syspath

PathLike = Union[str, Path]

ALDM_REPO_NAME = "aldm"
ALDM_LDM_CONFIG_RELPATH = "LDM/configs/latent-diffusion/brats-ldm-vq-4.yaml"
ALDM_VQGAN_STAGE1_CONFIG_RELPATH = "VQ-GAN/configs/brats_vqgan_stage1.yaml"
ALDM_VQGAN_STAGE2_CONFIG_RELPATH = "VQ-GAN/configs/brats_vqgan_stage2.yaml"


@dataclass(frozen=True)
class ALDMRepo:
    """Resolved ALDM repository with convenience accessors for known subtrees."""

    root: Path

    @property
    def vqgan_root(self) -> Path:
        return (self.root / "VQ-GAN").resolve()

    @property
    def ldm_root(self) -> Path:
        return (self.root / "LDM").resolve()

    def require_vqgan_root(self) -> Path:
        root = self.vqgan_root
        if not root.exists():
            raise FileNotFoundError(f"ALDM repo missing VQ-GAN/ at {root}")
        return root

    def require_ldm_root(self) -> Path:
        root = self.ldm_root
        if not root.exists():
            raise FileNotFoundError(f"ALDM repo missing LDM/ at {root}")
        return root

    def resolve(self, path: PathLike) -> Path:
        path_obj = Path(path).expanduser()
        if path_obj.is_absolute():
            return path_obj.resolve(strict=False)
        return (self.root / path_obj).resolve(strict=False)

    def resolve_existing(self, path: PathLike, *, label: str) -> Path:
        resolved = self.resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"{label} not found: {resolved}")
        return resolved

    def resolve_default_or_override(
        self,
        path: Optional[PathLike],
        *,
        default_relpath: PathLike,
        label: str,
    ) -> Path:
        """Resolve an optional ALDM config path.

        ``None`` uses the known upstream ALDM default. Relative paths are
        interpreted relative to the ALDM repo root, and absolute paths are used
        as-is. Paths are never resolved relative to the process working
        directory.
        """

        return self.resolve_existing(default_relpath if path in (None, "") else path, label=label)

    @property
    def default_ldm_config_path(self) -> Path:
        return self.resolve_existing(ALDM_LDM_CONFIG_RELPATH, label="ALDM LDM config")

    @property
    def default_vqgan_stage1_config_path(self) -> Path:
        return self.resolve_existing(ALDM_VQGAN_STAGE1_CONFIG_RELPATH, label="ALDM VQ-GAN stage-1 config")

    @property
    def default_vqgan_stage2_config_path(self) -> Path:
        return self.resolve_existing(ALDM_VQGAN_STAGE2_CONFIG_RELPATH, label="ALDM VQ-GAN stage-2 config")


def get_aldm_repo(*, external_repo_root: Optional[PathLike] = None, allow_network: bool = True) -> ALDMRepo:
    """Resolve the configured ALDM repository from the BrainMint registry.

    ALDM source location is intentionally not a Hydra model parameter. The
    registry and ExternalRepoManager define where the repo is materialized.
    If the repo is missing, ExternalRepoManager may materialize it from the
    registry source using its default policy.
    """

    spec = get_repo_spec(ALDM_REPO_NAME)
    root = ExternalRepoManager(external_repo_root=external_repo_root).ensure_repo(spec, allow_network=allow_network)
    return ALDMRepo(root=root.resolve())


@contextmanager
def aldm_repo_context(*, external_repo_root: Optional[PathLike] = None, allow_network: bool = True) -> Iterator[ALDMRepo]:
    """Yield a resolved ALDM repo without modifying ``sys.path``."""

    yield get_aldm_repo(external_repo_root=external_repo_root, allow_network=allow_network)


@contextmanager
def vqgan_import_context(*, external_repo_root: Optional[PathLike] = None, allow_network: bool = True) -> Iterator[ALDMRepo]:
    """Temporarily expose ALDM's ``VQ-GAN`` package root."""

    repo = get_aldm_repo(external_repo_root=external_repo_root, allow_network=allow_network)
    with repo_on_syspath([repo.require_vqgan_root()]):
        yield repo


@contextmanager
def ldm_import_context(*, external_repo_root: Optional[PathLike] = None, allow_network: bool = True) -> Iterator[ALDMRepo]:
    """Temporarily expose ALDM's ``LDM`` and ``VQ-GAN`` package roots."""

    repo = get_aldm_repo(external_repo_root=external_repo_root, allow_network=allow_network)
    with repo_on_syspath([repo.require_ldm_root(), repo.require_vqgan_root()]):
        yield repo

