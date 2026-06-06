from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGE_SAFE_PREFIXES = (
    "tests/package/",
    "tests/external/",
    "tests/utils/",
)


def _repo_relative(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        return path.as_posix()
    return rel.as_posix()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "package_safe: tests that run from the standalone package without study configs, GPU, checkpoints, or external repositories",
    )
    config.addinivalue_line(
        "markers",
        "study_config: tests that require the original study config tree or study scripts",
    )
    config.addinivalue_line(
        "markers",
        "optional_stack: tests that require optional BrainMint dependency extras",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-study-config-tests",
        action="store_true",
        default=False,
        help="Collect copied legacy tests that require study configs/scripts or optional heavy stacks.",
    )


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    if path.suffix != ".py":
        return False

    rel = _repo_relative(path)
    if rel.startswith(PACKAGE_SAFE_PREFIXES):
        return False

    return not config.getoption("--run-study-config-tests")


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        rel = _repo_relative(Path(str(item.fspath)))
        if rel.startswith(PACKAGE_SAFE_PREFIXES):
            item.add_marker(pytest.mark.package_safe)
        else:
            item.add_marker(pytest.mark.study_config)
            item.add_marker(pytest.mark.optional_stack)
