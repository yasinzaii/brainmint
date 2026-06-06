# BrainMint

BrainMint is a Python package for brain MRI synthesis, compression, modality translation, inference utilities, and evaluation support.

This repository is the reusable package extracted from the original GenMRI study repository. It should contain library code, package-safe tests, and package documentation only. Study configs, private paths, datasets, checkpoints, generated outputs, and cloned external repositories should stay outside this repository.

## Install

Install the base package with pip:

```bash
pip install brainmint
```

Common optional installs:

| Use case | Install command |
| --- | --- |
| Metrics and evaluation | `pip install "brainmint[metrics]"` |
| Training utilities | `pip install "brainmint[train]"` |
| Full runtime stack | `pip install "brainmint[all]"` |

Smaller targeted extras are kept in `pyproject.toml` for dependency control, but the commands above are the ones most users should need.

## External Repositories

External model repositories are materialized outside the package tree. For research runs, set:

```bash
export BRAINMINT_EXTERNAL_ROOT=/path/to/external
```

or configure it in Python:

```python
import brainmint.external

brainmint.external.set_external_repo_root("/path/to/external")
```

## Tests

Default tests are package-safe and do not require study Hydra configs, GPU, checkpoints, network access, or cloned external repositories:

```bash
python -m pytest
```

Copied study-era tests are opt-in while BrainMint is being separated from the original study repository:

```bash
python -m pytest --run-study-config-tests
```

## License

MIT. See `LICENSE`.
