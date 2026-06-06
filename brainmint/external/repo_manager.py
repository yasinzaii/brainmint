# brainmint/external/repo_manager.py
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

from .sys_path import repo_on_syspath, resolve_path

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

BRAINMINT_EXTERNAL_ROOT_ENV = "BRAINMINT_EXTERNAL_ROOT"
_EXTERNAL_REPO_ROOT_OVERRIDE: Optional[Path] = None


def set_external_repo_root(path: Optional[PathLike]) -> None:
    """Set the process-wide external repo root."""

    global _EXTERNAL_REPO_ROOT_OVERRIDE
    _EXTERNAL_REPO_ROOT_OVERRIDE = None if path is None else resolve_path(path)


def get_external_repo_root() -> Path:
    """Return the currently resolved external repo root."""
    return default_external_repo_root()


@dataclass(frozen=True)
class RepoSpec:
    """Description of an external repo materialized under external/repos/<repo_dirname>/."""
    name: str
    repo_dirname: str
    python_roots: Sequence[str] = (".",)

    # Optional local bootstrap zip
    zip_filename: Optional[str] = None

    # Optional network sources
    git_url: Optional[str] = None
    git_ref: Optional[str] = None  # branch/tag/commit
    strip_git_dir: bool = False

    hf_repo_id: Optional[str] = None
    hf_revision: Optional[str] = None


def _platform_user_cache_root() -> Path:
    """Return the platform-native cache root for installed-package use."""

    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return (resolve_path(base) / "brainmint" / "Cache").resolve()
        return (Path.home() / "AppData" / "Local" / "brainmint" / "Cache").resolve()

    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Caches" / "brainmint").resolve()

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return (resolve_path(xdg_cache_home) / "brainmint").resolve()
    return (Path.home() / ".cache" / "brainmint").resolve()


def default_external_repo_root() -> Path:
    """Resolve the default root that contains external repo materializations.

    Priority:
      1. Explicit process override set by :func:`set_external_repo_root`.
      2. ``BRAINMINT_EXTERNAL_ROOT``: direct path to the external repo root.
      3. Platform-native user cache fallback.
    """

    if _EXTERNAL_REPO_ROOT_OVERRIDE is not None:
        return _EXTERNAL_REPO_ROOT_OVERRIDE

    env_external = os.environ.get(BRAINMINT_EXTERNAL_ROOT_ENV)
    if env_external:
        return resolve_path(env_external)

    return (_platform_user_cache_root() / "external").resolve()


class ExternalRepoManager:
    def __init__(
        self,
        *,
        external_repo_root: Optional[PathLike] = None,
        lock_timeout_s: int = 600,
        lock_poll_s: float = 0.5,
    ) -> None:
        self.external_repo_root = (
            resolve_path(external_repo_root)
            if external_repo_root is not None
            else default_external_repo_root()
        )
        self.repos_root = (self.external_repo_root / "repos").resolve()
        self.lock_timeout_s = int(lock_timeout_s)
        self.lock_poll_s = float(lock_poll_s)

    def repo_path(self, spec: RepoSpec) -> Path:
        return (self.repos_root / spec.repo_dirname).resolve()

    def ensure_repo(
        self,
        spec: RepoSpec,
        *,
        overwrite: bool = False,
        allow_network: bool = True,
        delete_zip_after_extract: bool = True,
    ) -> Path:
        """Ensure repo exists on disk; auto-materialize if missing."""
        self.repos_root.mkdir(parents=True, exist_ok=True)
        repo_dir = self.repo_path(spec)

        with self._repo_lock(spec.repo_dirname):
            if overwrite and repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)

            if self._looks_ready(repo_dir):
                return repo_dir

            # 1) local zip bootstrap (optional)
            zip_path = self._find_zip(spec)
            if zip_path is not None:
                log.info("Bootstrapping external repo '%s' from zip: %s", spec.name, zip_path)
                self._extract_zip(zip_path, repo_dir)
                if delete_zip_after_extract:
                    try:
                        zip_path.unlink()
                    except Exception:
                        pass
                if self._looks_ready(repo_dir):
                    return repo_dir

            # 2) git clone
            if allow_network and spec.git_url:
                log.info("Fetching external repo '%s' via git clone: %s", spec.name, spec.git_url)
                self._clone_git(spec.git_url, repo_dir, ref=spec.git_ref, strip_git_dir=spec.strip_git_dir)
                if self._looks_ready(repo_dir):
                    return repo_dir

            # 3) HuggingFace snapshot
            if allow_network and spec.hf_repo_id:
                log.info("Fetching external repo '%s' via HuggingFace snapshot: %s", spec.name, spec.hf_repo_id)
                self._snapshot_hf(spec.hf_repo_id, repo_dir, revision=spec.hf_revision)
                if self._looks_ready(repo_dir):
                    return repo_dir

            raise FileNotFoundError(
                f"External repo '{spec.name}' not available at {repo_dir}. "
                f"Provide it manually, call brainmint.external.set_external_repo_root(...), set {BRAINMINT_EXTERNAL_ROOT_ENV}, "
                "pass external_repo_root, or set RepoSpec.git_url / hf_repo_id."
            )

    def _looks_ready(self, repo_dir: Path) -> bool:
        if not repo_dir.exists() or not repo_dir.is_dir():
            return False
        try:
            next(repo_dir.iterdir())
            return True
        except StopIteration:
            return False

    def _find_zip(self, spec: RepoSpec) -> Optional[Path]:
        candidates = []
        if spec.zip_filename:
            candidates += [
                self.external_repo_root / spec.zip_filename,
                self.external_repo_root / "zips" / spec.zip_filename,
            ]
        candidates += [
            self.external_repo_root / f"{spec.repo_dirname}.zip",
            self.external_repo_root / "zips" / f"{spec.repo_dirname}.zip",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    def _extract_zip(self, zip_path: Path, repo_dir: Path) -> None:
        tmp = repo_dir.with_name(repo_dir.name + f".tmp-{uuid.uuid4().hex}")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)

        children = [p for p in tmp.iterdir() if p.name != "__MACOSX"]
        extracted_root = children[0] if len(children) == 1 and children[0].is_dir() else tmp

        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        extracted_root.replace(repo_dir)
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    def _clone_git(self, url: str, repo_dir: Path, *, ref: Optional[str], strip_git_dir: bool) -> None:
        tmp = repo_dir.with_name(repo_dir.name + f".tmp-{uuid.uuid4().hex}")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

        shallow = True
        if ref and self._looks_like_commit(ref):
            shallow = False

        cmd = ["git", "clone"]
        if shallow and ref:
            cmd += ["--depth", "1", "--branch", ref]
        elif shallow:
            cmd += ["--depth", "1"]
        cmd += [url, str(tmp)]
        self._run(cmd)

        if ref and not shallow:
            self._run(["git", "-C", str(tmp), "checkout", ref])

        if strip_git_dir:
            gd = tmp / ".git"
            if gd.exists():
                shutil.rmtree(gd, ignore_errors=True)

        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        tmp.replace(repo_dir)

    def _snapshot_hf(self, repo_id: str, repo_dir: Path, *, revision: Optional[str]) -> None:
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required for hf_repo_id fetching. Install it or use git_url."
            ) from e

        tmp = repo_dir.with_name(repo_dir.name + f".tmp-{uuid.uuid4().hex}")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)

        cache_path = Path(snapshot_download(repo_id=repo_id, revision=revision))
        shutil.copytree(cache_path, tmp, dirs_exist_ok=True)

        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        tmp.replace(repo_dir)

    def _run(self, cmd: Sequence[str]) -> None:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

    def _looks_like_commit(self, ref: str) -> bool:
        r = ref.strip()
        if len(r) < 7 or len(r) > 40:
            return False
        return all(ch in "0123456789abcdefABCDEF" for ch in r)

    def _lock_dir(self, repo_dirname: str) -> Path:
        return self.repos_root / f".{repo_dirname}.lock"

    @contextmanager
    def _repo_lock(self, repo_dirname: str) -> Iterator[None]:
        lock_dir = self._lock_dir(repo_dirname)
        start = time.time()
        while True:
            try:
                lock_dir.mkdir(parents=False, exist_ok=False)
                break
            except FileExistsError:
                if time.time() - start > self.lock_timeout_s:
                    raise TimeoutError(f"Timed out waiting for lock {lock_dir}")
                time.sleep(self.lock_poll_s)
        try:
            yield
        finally:
            shutil.rmtree(lock_dir, ignore_errors=True)


@contextmanager
def import_external_repo(
    spec: RepoSpec,
    *,
    repo_root: Optional[Union[str, Path]] = None,
    external_repo_root: Optional[Union[str, Path]] = None,
    allow_network: bool = True,
    overwrite: bool = False,
    delete_zip_after_extract: bool = True,
) -> Iterator[Path]:
    """Activate an external repo on sys.path, auto-fetching if missing.

    ``repo_root`` is a hint to an already-materialized repo directory. If it is
    absent or missing, the manager resolves ``external_repo_root`` using
    :func:`default_external_repo_root` and materializes ``external_repo_root/repos/<repo>``.
    """
    # If repo_root exists, use it directly; otherwise ignore it (common when Hydra cwd changes).
    rr: Optional[Path] = resolve_path(repo_root) if repo_root is not None else None
    if rr is None or not rr.exists():
        if rr is not None and not rr.exists():
            log.warning(
                "repo_root='%s' does not exist for '%s'; falling back to external_repo_root='%s'.",
                rr, spec.name, external_repo_root or default_external_repo_root()
            )
        mgr = ExternalRepoManager(external_repo_root=external_repo_root)
        rr = mgr.ensure_repo(
            spec,
            overwrite=overwrite,
            allow_network=allow_network,
            delete_zip_after_extract=delete_zip_after_extract,
        )

    import_roots = [(rr / p).resolve() for p in spec.python_roots]
    for r in import_roots:
        if not r.exists():
            raise FileNotFoundError(f"Missing python root '{r}' for external repo '{spec.name}'.")

    with repo_on_syspath(import_roots):
        yield rr
