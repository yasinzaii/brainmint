"""Utilities for mapping stream dictionaries to explicit modality keys."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from monai.transforms import Transform

logger = logging.getLogger(__name__)

import multiprocessing as mp
from dataclasses import dataclass


@dataclass
class SharedSamplingState:
    """State shared across DataLoader workers for partial sampling config (optional)."""

    config: Dict[str, Any]
    shared: bool = True

    def __post_init__(self) -> None:
        if self.shared:
            self._mgr = mp.Manager()
            self._store = self._mgr.dict()
            self._store["config"] = dict(self.config)
        else:
            self._mgr = None
            self._store = {"config": dict(self.config)}

    def __getstate__(self) -> Dict[str, Any]:
        # IMPORTANT:
        # - When shared=True, keep the manager proxy (_store) so workers see updates.
        # - Never try to pickle the Manager object itself (_mgr).
        if self.shared:
            return {
                "shared": True,
                "_store": self._store,
                "config": {},
            }
        return {"config": self.get_config(), "shared": False}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.shared = bool(state.get("shared", False))
        self._mgr = None
        if self.shared:
            self._store = state["_store"]
        else:
            self._store = {"config": dict(state.get("config", {}))}
        self.config = dict(self.get_config())
    
    def get_config(self) -> Dict[str, Any]:
        return dict(self._store["config"])

    def set_config(self, config: Mapping[str, Any]) -> None:
        logger.info("SharedSamplingState.set_config called with config=%s", config)
        print(f"[SamplingState] set_config config={config}")
        self._store["config"] = dict(config)


def _lower_list(values: Sequence[str]) -> list[str]:
    return [str(v).lower() for v in values]


def _lookup_stream_value(stream_dict: Any, mod: str, *, allow_missing: bool) -> Optional[Any]:
    if not isinstance(stream_dict, Mapping):
        if allow_missing:
            return None
        raise TypeError(f"Expected mapping for stream data, got {type(stream_dict)}.")
    mod_keys = {str(k).lower(): k for k in stream_dict.keys()}
    key = mod_keys.get(mod)
    if key is None:
        if allow_missing:
            return None
        raise KeyError(f"Missing modality '{mod}' in stream mapping.")
    return stream_dict[key]


class MapChosenStreamToKeysd(Transform):
    """Map stream dictionaries into explicit keys using ``stream_key_map``."""

    def __init__(
        self,
        *,
        modalities: Sequence[str],
        stream_key_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        self.modalities = _lower_list(modalities)
        self.stream_key_map = {
            str(stream).lower(): {
                str(mod).lower(): str(key)
                for mod, key in dict(mods).items()
            }
            for stream, mods in dict(stream_key_map or {}).items()
        }
        self.allow_missing_keys = bool(allow_missing_keys)

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if not self.stream_key_map:
            return d

        for stream_name, stream_map in self.stream_key_map.items():
            stream_dict = d.get(stream_name)
            if not isinstance(stream_dict, Mapping):
                if not self.allow_missing_keys:
                    raise TypeError(f"Missing or invalid stream '{stream_name}' for mapping.")
                continue
            for mod, key_spec in stream_map.items():
                if not key_spec:
                    raise KeyError(f"Missing target key for modality '{mod}' in stream '{stream_name}'.")
                value = _lookup_stream_value(stream_dict, mod, allow_missing=self.allow_missing_keys)
                if value is None:
                    continue
                d[str(key_spec)] = value

        
        # Popping The Streams as no longer required.
        for stream_name, _ in self.stream_key_map.items():
            if stream_name in d:
                d.pop(stream_name, None)

        return d


class SampleLatentsFromKeysd(Transform):
    """Sample latent tensors from explicit mu/sigma keys after loading."""

    def __init__(
        self,
        *,
        modalities: Sequence[str],
        mu_key_map: Optional[Mapping[str, str]] = None,
        sigma_key_map: Optional[Mapping[str, str]] = None,
        out_key_map: Optional[Mapping[str, str]] = None,
        partial_sampling: Optional[Mapping[str, Any]] = None,
        partial_sampling_enabled: bool = True,
        sampling_state: Optional[SharedSamplingState] = None,
        require_sigma_for_sampling: bool = False,
        drop_input_keys: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        self.modalities = _lower_list(modalities)
        self.mu_key_map = {str(k).lower(): str(v) for k, v in dict(mu_key_map or {}).items()}
        self.sigma_key_map = {str(k).lower(): str(v) for k, v in dict(sigma_key_map or {}).items()}
        self.out_key_map = {str(k).lower(): str(v) for k, v in dict(out_key_map or {}).items()}
        self.partial_sampling = dict(partial_sampling or {})
        self.partial_sampling_enabled = bool(partial_sampling_enabled)
        self.require_sigma_for_sampling = bool(require_sigma_for_sampling)
        self.drop_input_keys = bool(drop_input_keys)
        self.allow_missing_keys = bool(allow_missing_keys)

        if sampling_state is not None:
            self.sampling_state = sampling_state
        else:
            self.sampling_state = SharedSamplingState(config=dict(self.partial_sampling), shared=True)

    def _get_partial_sampling(self) -> Dict[str, Any]:
        return self.sampling_state.get_config()

    def _apply_partial_sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        if not self.partial_sampling_enabled:
            return z_mu

        cfg = self._get_partial_sampling()
        p = float(cfg.get("sigma_prob", 0.0))
        alpha = float(cfg.get("sigma_alpha", 0.0))
        if p <= 0.0 or alpha == 0.0:
            return z_mu

        eps = torch.randn_like(z_mu)
        if z_mu.dim() == 5:
            mask_shape = (z_mu.size(0), 1, 1, 1, 1)
        elif z_mu.dim() == 4:
            mask_shape = (1, 1, 1, 1)
        else:
            raise ValueError(
                f"Expected 4D or 5D latent tensor for sampling, got shape {tuple(z_mu.shape)}."
            )

        mask = (torch.rand(mask_shape, device=z_mu.device) < p).float()
        return z_mu + mask * (alpha * z_sigma * eps)

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for idx, mod in enumerate(self.modalities):
            mu_key = self.mu_key_map.get(mod)
            sigma_key = self.sigma_key_map.get(mod)
            out_key = self.out_key_map.get(mod)
            if out_key is None:
                raise KeyError(f"Missing out_key_map entry for modality '{mod}'.")

            if mu_key not in d:
                if not self.allow_missing_keys:
                    raise KeyError(f"Missing mu key '{mu_key}' for modality '{mod}'.")
                continue

            mu_val = d[mu_key]
            if not torch.is_tensor(mu_val):
                if not self.allow_missing_keys:
                    raise TypeError(f"Expected tensor for '{mu_key}', got {type(mu_val)}.")
                continue

            sigma_val = d.get(sigma_key)
            if not torch.is_tensor(sigma_val):
                if self.require_sigma_for_sampling:
                    raise KeyError(f"Missing sigma tensor for modality '{mod}' at key '{sigma_key}'.")
                d[out_key] = mu_val
                if self.drop_input_keys:
                    if mu_key in d and mu_key != out_key:
                        d.pop(mu_key, None)
                    if sigma_key in d and sigma_key != out_key:
                        d.pop(sigma_key, None)
                continue

            d[out_key] = self._apply_partial_sampling(mu_val, sigma_val)
            if self.drop_input_keys:
                if mu_key in d and mu_key != out_key:
                    d.pop(mu_key, None)
                if sigma_key in d and sigma_key != out_key:
                    d.pop(sigma_key, None)

        return d
