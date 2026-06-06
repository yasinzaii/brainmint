from __future__ import annotations


def test_import_brainmint_has_version() -> None:
    import brainmint

    assert isinstance(brainmint.__version__, str)
    assert brainmint.__version__
