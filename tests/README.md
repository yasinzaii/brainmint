# BrainMint Tests

> BrainMint was split out from the original GenMRI/Synthetic MRI study repository. Many copied tests still reflect that study repo: they may expect Hydra config trees, training scripts, local datasets, checkpoints, external repositories, or GPU access. Those tests are kept temporarily as migration references and need to be rewritten into package-owned tests.

## Default Tests

Run the package-safe test suite from the repository root:

```bash
python -m pytest
```

These are the tests that should pass for the standalone BrainMint package. They avoid private paths, study configs, checkpoints, network access, and GPU requirements.

Current package-safe test areas:

- `tests/package/`: import and public package surface checks.
- `tests/external/`: external repository root and manager behavior without fetching real repositories.
- `tests/utils/`: small utility behavior.

You can run one area directly:

```bash
python -m pytest tests/package
python -m pytest tests/external
python -m pytest tests/utils
```

## Legacy Study Tests

The copied study-era tests remain in their original folders while the package split is finished. They are intentionally skipped by default.

Run them explicitly with:

```bash
python -m pytest --run-study-config-tests
```

Run a single legacy file with:

```bash
python -m pytest --run-study-config-tests tests/data/test_brainscape.py
```

Warning: these tests may fail in this standalone package because the original GenMRI Hydra configs, local scripts, datasets, checkpoints, or external repositories are not package-owned BrainMint test fixtures. Treat failures in these tests as migration work, not as default package regressions.

