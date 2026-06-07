"""BraSyn / BrainLesion MissingMRI modality mapping helpers."""

from __future__ import annotations

from collections.abc import Mapping

GM_TO_BRAKEY = {
    "t1w": "t1n",
    "t1ce": "t1c",
    "t2w": "t2w",
    "flair": "t2f",
}

CANONICAL_MODALITIES = ("t1w", "t2w", "flair", "t1ce")


def _bra_key(modality: str) -> str:
    modality = str(modality).lower()
    if modality not in GM_TO_BRAKEY:
        raise KeyError(f"Unknown modality {modality!r} (expected one of {sorted(GM_TO_BRAKEY)})")
    return GM_TO_BRAKEY[modality]


def build_missingmri_infer_kwargs(
    *,
    target: str,
    cond_paths: Mapping[str, str | None],
    zero_path: str,
) -> dict[str, str]:
    """Build keyword arguments for ``MissingMRI.infer_single``."""

    target = str(target).lower()
    if target not in GM_TO_BRAKEY:
        raise ValueError(f"Unsupported target modality {target!r}; expected one of {sorted(GM_TO_BRAKEY)}")

    kwargs: dict[str, str] = {}
    for modality in CANONICAL_MODALITIES:
        if modality == target:
            continue
        path = cond_paths.get(modality)
        kwargs[GM_TO_BRAKEY[modality]] = path if path else zero_path
    return kwargs


def _resolve_case_inputs(
    case: str,
    target: str,
    real_paths: Mapping[str, str],
    syn_paths: Mapping[str, str],
) -> dict[str, str]:
    """Pick modality file paths to feed for a metrics/inference completion case."""

    case = str(case).lower()
    target = str(target).lower()

    def pick(modality: str) -> str | None:
        if case in {"brainscape", "brainscape_all_completion"}:
            return real_paths.get(modality) or syn_paths.get(modality)
        if case in {"brats", "brats_completion"}:
            return real_paths.get(modality)
        if case in {"brats_t1w_only", "t1w_only"}:
            return syn_paths.get(modality)
        raise ValueError(f"Unknown completion case {case!r}")

    inputs: dict[str, str] = {}
    for modality in CANONICAL_MODALITIES:
        path = pick(modality)
        if path:
            inputs[modality] = path
    inputs.pop(target, None)
    return inputs
