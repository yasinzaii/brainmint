"""BrainMint models, integrations, and inference tools for medical image synthesis."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError):
    try:
        __version__ = _metadata_version("brainmint")
    except PackageNotFoundError:
        __version__ = "0.0.0"
