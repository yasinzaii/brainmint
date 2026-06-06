from __future__ import annotations

from typing import Dict

from .repo_manager import RepoSpec

# External model repositories managed outside the BrainMint package.
REPOS: Dict[str, RepoSpec] = {

    "ha_gan": RepoSpec(
        name="ha_gan",
        repo_dirname="ha_gan",
        python_roots=(".",),
        git_url="https://github.com/batmanlab/HA-GAN.git",
        git_ref=None,
    ),
    "medicaldiffusion": RepoSpec(
        name="medicaldiffusion",
        repo_dirname="medicaldiffusion",
        python_roots=(".",),
        git_url="https://github.com/FirasGit/medicaldiffusion.git",
        git_ref=None,
    ),
    "wdm_3d": RepoSpec(
        name="wdm_3d",
        repo_dirname="wdm_3d",
        python_roots=(".",),
        git_url="https://github.com/pfriedri/wdm-3d.git",
        git_ref=None,
    ),
    "medsyn": RepoSpec(
        name="medsyn",
        repo_dirname="medsyn",
        # MedSyn keeps its python package under src/
        python_roots=("src",),
        git_url="https://github.com/batmanlab/MedSyn.git",
        git_ref=None,
    ),

    "med_ddpm": RepoSpec(
        name="med_ddpm",
        repo_dirname="med_ddpm",
        python_roots=(".",),
        git_url="https://github.com/mobaidoctor/med-ddpm",
        git_ref=None,
    ),

    "aldm": RepoSpec(
        name="aldm",
        repo_dirname="aldm",
        python_roots=("LDM", "VQ-GAN"),
        git_url="https://github.com/jongdory/ALDM.git",
        git_ref=None,
    ),
    
    "cwdm": RepoSpec(
        name="cwdm",
        repo_dirname="cwdm",
        python_roots=(".",),
        git_url="https://github.com/pfriedri/cwdm.git",
        git_ref=None,
    ),

    "maisi": RepoSpec(
        name="maisi",
        repo_dirname="nv_generate_mr",
        python_roots=(".",),
        git_url="https://github.com/NVIDIA-Medtech/NV-Generate-CTMR.git",
        git_ref=None,
    ),
}


def get_repo_spec(name: str) -> RepoSpec:
    key = str(name)
    if key not in REPOS:
        raise KeyError(f"Unknown external repo '{key}'. Available: {sorted(REPOS.keys())}")
    return REPOS[key]
