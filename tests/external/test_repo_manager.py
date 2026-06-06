from __future__ import annotations

from pathlib import Path

import pytest

import brainmint.external as external
from brainmint.external.repo_manager import (
    BRAINMINT_EXTERNAL_ROOT_ENV,
    ExternalRepoManager,
    RepoSpec,
    default_external_repo_root,
    set_external_repo_root,
)


@pytest.fixture(autouse=True)
def _reset_external_repo_root() -> None:
    set_external_repo_root(None)
    yield
    set_external_repo_root(None)


def test_external_root_env_uses_brainmint_name(monkeypatch, tmp_path: Path) -> None:
    set_external_repo_root(None)
    monkeypatch.setenv(BRAINMINT_EXTERNAL_ROOT_ENV, str(tmp_path))

    assert default_external_repo_root() == tmp_path.resolve()


def test_explicit_external_root_override_wins(monkeypatch, tmp_path: Path) -> None:
    env_root = tmp_path / "env"
    override_root = tmp_path / "override"
    monkeypatch.setenv(BRAINMINT_EXTERNAL_ROOT_ENV, str(env_root))

    set_external_repo_root(override_root)

    assert external.get_external_repo_root() == override_root.resolve()


def test_repo_path_is_under_repos_root(tmp_path: Path) -> None:
    mgr = ExternalRepoManager(external_repo_root=tmp_path)
    spec = RepoSpec(name="demo", repo_dirname="demo-repo")

    assert mgr.repo_path(spec) == (tmp_path / "repos" / "demo-repo").resolve()
