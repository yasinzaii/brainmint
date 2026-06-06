import importlib
import importlib.util


def test_lightweight_package_roots_do_not_reexport_model_wrappers():
    compression = importlib.import_module("brainmint.models.compression")
    generation = importlib.import_module("brainmint.models.generation")
    translation = importlib.import_module("brainmint.models.translation")

    assert compression.__all__ == []
    assert generation.__all__ == []
    assert translation.__all__ == []

    assert not hasattr(compression, "LDMVAE")
    assert not hasattr(generation, "WDM3DWrapper")
    assert not hasattr(translation, "ALDMModalityTranslator")


def test_external_package_root_exposes_only_public_root_api():
    external = importlib.import_module("brainmint.external")

    assert sorted(external.__all__) == [
        "BRAINMINT_EXTERNAL_ROOT_ENV",
        "ExternalRepoManager",
        "get_external_repo_root",
        "set_external_repo_root",
    ]

    for name in ("RepoSpec", "default_external_repo_root", "get_repo_spec", "import_external_repo"):
        assert not hasattr(external, name)


def test_concrete_submodule_imports_remain_available():
    from brainmint.external.registry import get_repo_spec
    from brainmint.external.repo_manager import RepoSpec, default_external_repo_root, import_external_repo

    assert get_repo_spec is not None
    assert RepoSpec is not None
    assert default_external_repo_root is not None
    assert import_external_repo is not None
    assert importlib.util.find_spec("brainmint.models.generation.diffusion_unet") is not None
    assert importlib.util.find_spec("brainmint.models.compression.ldm_vae") is not None
    assert importlib.util.find_spec("brainmint.models.translation.aldm") is not None
