import os
import logging
import inspect

from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Union

import torch
from torch import nn

_LOG = logging.getLogger(__name__)


def _safe_load(path: str, map_location: Any = "cpu") -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # PyTorch >= 2.0
    except Exception as e:
        _LOG.warning("weights_only=True failed for %s (%s); falling back to weights_only=False", path, e)
        return torch.load(path, map_location=map_location)

def get_attr(obj: Any, dot_path: str) -> Any:
    """Resolve a dotted attribute path on an object."""

    cur = obj
    for part in dot_path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(f"Target path '{dot_path}' not found at '{part}'")
        cur = getattr(cur, part)
    return cur


def _flatten_map(m: Mapping[str, Any], prefix: str = "") -> Dict[str, torch.Tensor]:
    """Flatten nested mappings into {'a.b.c': Tensor, ...}. Ignore non-tensor leaves."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in m.items():
        if not isinstance(k, str):
            continue
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping):
            out.update(_flatten_map(v, key))
        else:
            if isinstance(v, torch.Tensor):
                out[key] = v
    return out

def _slice_by_prefix(sd: Mapping[str, Any], prefix: str) -> Dict[str, torch.Tensor]:
    """Slice a (possibly nested) mapping by dotted prefix; strip that prefix."""
    flat = _flatten_map(sd)
    if not prefix:
        return dict(flat)
    pref = prefix if prefix.endswith(".") else prefix + "."
    plen = len(pref)
    out: Dict[str, torch.Tensor] = {}
    for k, v in flat.items():
        if k.startswith(pref):
            out[k[plen:]] = v
    return out


def _find_state_dict(container: Mapping[str, Any], locator: Optional[str]) -> Mapping[str, torch.Tensor]:
    """
    Return a UNIQUE slice of weights using a single locator string.

    Locator examples:
      - "state_dict.autoencoder.encoder"
      - "autoencoder.encoder"
      - "state_dict"

      - "<root>" or "." - Will skip finding locator,
    """

    if not isinstance(container, Mapping):
        raise ValueError("Checkpoint container is not a mapping.")
    if locator is None or not isinstance(locator, str):
        raise ValueError("Locator must be a non-empty string (use target or state_key).")

    bases: List[tuple[Optional[str], Mapping[str, Any]]] = []
    for cont_name in ("state_dict", "model"):
        b = container.get(cont_name)
        if isinstance(b, Mapping):
            bases.append((cont_name, b))
    bases.append((None, container))

    # Normalize explicit root requests ("."/"<root>"); empty string already works
    if locator in (".", "<root>"):
        locator = ""

    # Build candidate slices
    candidates: List[Dict[str, torch.Tensor]] = []
    for tag, base in bases:
        if tag is not None and locator == tag:
            cand = _flatten_map(base)
            if cand:
                candidates.append(cand)
            continue
        cand = _slice_by_prefix(base, locator)
        if cand:
            candidates.append(cand)

    # Deduplicate by key signature (avoid double-counting identical slices)
    uniq: Dict[tuple, Dict[str, torch.Tensor]] = {}
    for c in candidates:
        sig = tuple(sorted(c.keys()))
        if sig:
            uniq.setdefault(sig, c)

    if len(uniq) == 1:
        return next(iter(uniq.values()))
    elif len(uniq) == 0:
        raise ValueError(f"No matching weights for locator '{locator}'. "
                          "Checked under: ['state_dict', 'model', '<root>'].")
    else:
        raise ValueError(f"Ambiguous locator '{locator}': matched {len(uniq)} distinct slices. "
                          "Clearify by using a longer locator (e.g., include 'state_dict.').")

def load_module_state_dict(
    module: nn.Module,
    *,
    path: str,
    state_key: Optional[str],
    strict: Union[bool, str] = "auto",
    loader: Optional[str] = None,
    freeze: bool = True,
    set_eval: bool = True,
    optional: bool = False,
    target_name: str = "<module>",
) -> bool:
    """
    Load one checkpoint slice into a concrete module.

    StateDictLoaderMixin resolves target names on an owning object. This
    function is the reusable implementation for callers that already have the
    target module.
    """
    path = os.fspath(path)
    if isinstance(strict, str):
        strict_lower = strict.lower()
        if strict_lower == "auto":
            strict_mode: Union[bool, str] = "auto"
        elif strict_lower in {"true", "1", "yes"}:
            strict_mode = True
        elif strict_lower in {"false", "0", "no"}:
            strict_mode = False
        else:
            raise ValueError(f"Unsupported strict value for target='{target_name}': {strict!r}")
    else:
        strict_mode = bool(strict)

    if loader is None:
        ckpt_loader = module.load_state_dict
        loader_supports_strict = True
        loader_name = "load_state_dict"
    else:
        ckpt_loader = get_attr(module, str(loader))
        if not callable(ckpt_loader):
            raise TypeError(f"loader '{loader}' on target '{target_name}' is not callable")
        try:
            loader_supports_strict = "strict" in inspect.signature(ckpt_loader).parameters
        except (TypeError, ValueError):
            loader_supports_strict = False
        loader_name = str(loader)

    if not os.path.exists(path):
        if optional:
            _LOG.warning("Optional checkpoint missing for target='%s': %s (skipping)", target_name, path)
            return False
        raise FileNotFoundError(f"Checkpoint not found for target='{target_name}': {path}")

    ckpt = _safe_load(path, map_location="cpu")
    if not isinstance(ckpt, Mapping):
        raise ValueError(f"Checkpoint at {path} is not a mapping; got {type(ckpt)}")

    locator = state_key if state_key is not None else target_name
    sd = _find_state_dict(ckpt, locator)

    if strict_mode == "auto":
        try:
            if loader_supports_strict:
                result = ckpt_loader(sd, strict=True)
            else:
                result = ckpt_loader(sd)
        except RuntimeError as e:
            _LOG.warning("Strict load failed for '%s' (%s). Retrying with shape-filter + strict=False.", target_name, e)
            sd2 = _shape_filter(sd, module)
            if loader_supports_strict:
                result = ckpt_loader(sd2, strict=False)
            else:
                result = ckpt_loader(sd2)
    elif strict_mode is True:
        if loader_supports_strict:
            result = ckpt_loader(sd, strict=True)
        else:
            result = ckpt_loader(sd)
    else:
        sd2 = _shape_filter(sd, module)
        if loader_supports_strict:
            result = ckpt_loader(sd2, strict=False)
        else:
            result = ckpt_loader(sd2)

    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    _LOG.log(
        logging.WARNING if (missing or unexpected) else logging.INFO,
        "Loaded target='%s' from %s using %s (missing=%d [%s]; unexpected=%d [%s])",
        target_name,
        path,
        loader_name,
        len(missing),
        _peek(missing),
        len(unexpected),
        _peek(unexpected),
    )

    if freeze:
        for p in module.parameters():
            p.requires_grad = False
    if set_eval:
        module.eval()

    return True


def _load_weight_spec(owner: Any, spec: Mapping[str, Any]) -> Optional[str]:
    """Load one normalized weight spec; used by load_weight_specs()."""

    spec = dict(spec)
    target = str(spec["target"])
    try:
        module = get_attr(owner, target)
    except AttributeError as exc:
        raise RuntimeError(f"Spec target not found: {target}") from exc
    if not isinstance(module, nn.Module):
        raise TypeError(f"Spec target '{target}' is not an nn.Module")

    loaded = load_module_state_dict(
        module,
        path=str(spec["path"]),
        state_key=spec.get("state_key"),
        strict=spec.get("strict", "auto"),
        loader=spec.get("loader", None),
        freeze=bool(spec.get("freeze", True)),
        set_eval=bool(spec.get("eval", spec.get("set_eval", True))),
        optional=bool(spec.get("optional", False)),
        target_name=target,
    )
    return target if loaded else None


def load_weight_specs(owner: Any, specs: Iterable[Mapping[str, Any]]) -> List[str]:
    """Load Hydra-style weight specs into modules owned by ``owner``.

    ``owner`` should be the wrapper/container object that owns the modules named
    by the specs. Targets are resolved through attribute lookup. For example,
    ``target: "autoencoder"`` loads into ``owner.autoencoder`` and
    ``target: "model.encoder"`` loads into ``owner.model.encoder``.

    Every loadable model must therefore be attached as a member on ``owner``
    before calling this function. If a runner receives modules in a dict, it
    should also register them as attributes first.

    Example:
        class Wrapper(nn.Module):
            def __init__(self, autoencoder: nn.Module, unet: nn.Module) -> None:
                super().__init__()
                self.autoencoder = autoencoder
                self.unet = unet

        loaded = load_weight_specs(wrapper, [
            {"target": "autoencoder", "path": vae_ckpt, "state_key": "autoencoder"},
            {"target": "unet", "path": unet_ckpt, "state_key": "unet"},
        ])

    Required spec keys are ``target`` and ``path``. Optional keys are forwarded
    to :func:`load_module_state_dict`: ``state_key``, ``strict``, ``loader``,
    ``freeze``, ``eval``/``set_eval``, and ``optional``.

    Exceptions are wrapped with the spec index, so failures point back to the
    exact entry in the list. Returns loaded target names in spec order.
    Duplicate targets are preserved.
    """

    loaded_targets: List[str] = []
    for index, spec in enumerate(specs):
        spec = dict(spec)
        try:
            target = _load_weight_spec(owner, spec)
        except Exception as exc:
            raise RuntimeError(f"Failed loading weight spec #{index}: {spec}") from exc
        if target is not None:
            loaded_targets.append(target)
    return loaded_targets


def _shape_filter(sd: Mapping[str, torch.Tensor], module: nn.Module) -> Dict[str, torch.Tensor]:
    """Drop keys whose tensor shape doesn't match the target module."""
    tgt = module.state_dict()
    return {k: v for k, v in sd.items() if (k in tgt and tuple(tgt[k].shape) == tuple(v.shape))}

def _peek(names: Iterable[str], n: int = 8) -> str:
    names = list(names)
    return ", ".join(names[:n]) + (" ..." if len(names) > n else "")

def _trainer_is_restoring(trainer: Any) -> bool:
    """Public Lightning surface: 'ckpt_path' is set during restore (fit/validate/test/predict)."""
    if not trainer:
        return False
    else:
        return bool(getattr(trainer, "ckpt_path", None))


class StateDictLoaderMixin:
    """
    StateDictLoaderMixin — Weight loader for a target module.

    Usage:
        class MyModule(StateDictLoaderMixin, pl.LightningModule): ...
        # put a list of specs under self.hparams["weight_loads"]

    Spec (per target):
    - target (str, required): submodule of LightningModule (e.g. "unet", "autoencoder.encoder")
        - path   (str, required): checkpoint file (.ckpt/.pt/.pth)
            - state_key (str|None, default=None): locator for weights inside the ckpt.
                * nested path or flat prefix, e.g. "state_dict.autoencoder.encoder" or "autoencoder.encoder"
                * searched under: container["state_dict"] -> ["model"] -> root
                * must match **exactly one** slice; else error

                * use "<root>" (or ".") for flat checkpoints whose weights live at the top level

            - strict ("auto"|True|False, default="auto"): strict load or shape-filtered fallback
            - freeze (bool, default=True): set requires_grad=False after load
            - eval (bool, default=True): call .eval() after load
            - optional (bool, default=False): if file missing, warn+skip
            - loader (str|None, default=None): optional module method name to use instead of load_state_dict

    Behavior:
    - Runs once in setup(stage) for fit/validate/test/predict.
    - If Lightning is restoring (trainer.ckpt_path set), it **skips** custom loads.
    - Logs missing/unexpected keys; applies freeze/eval if requested.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._loaded_once = False
        self._loaded_targets: Set[str] = set()

    def _specs(self) -> List[Dict[str, Any]]:
        hp = getattr(self, "hparams", {}) or {}
        specs = hp.get("weight_loads", []) or []
        return [dict(s) for s in specs]

    # Lightning hook: runs at start of fit/validate/test/predict
    def setup(self, stage: Optional[str] = None) -> None:
        if hasattr(super(), "setup"):
            super().setup(stage)  # Lightning setup()

        # If Lightning is restoring from a checkpoint, Skip custom loads.
        has_trainer = getattr(self, "_trainer", None) is not None
        if has_trainer:
            if _trainer_is_restoring(getattr(self, "trainer", None)):
                _LOG.info("trainer.ckpt_path is set; skipping custom state_dict loads.")
                self._loaded_once = True
                return

        if self._loaded_once:
            return

        self._loaded_targets.update(load_weight_specs(self, self._specs()))
        self._loaded_once = True

    def get_loaded_models(self) -> List[str]:
        """Return a list of top-level targets whose weights were loaded."""
        return sorted(self._loaded_targets)
