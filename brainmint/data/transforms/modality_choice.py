# brainmint/data/transforms/modality_choice.py
"""brainmint.data.transforms.modality_choice

Pick which *stream* (image/synthetic/latent/...) to use for each modality in a
multi-modal BrainScape sample.

This is intended to run **before** MONAI LoadImaged.

Input sample shape (from BrainScapePairedDataModule):

    {
      "image": {"t1w": "/.../t1w.nii.gz", "t2w": "/.../t2w.nii.gz"},
      "synthetic": {"t2w": "/.../t2w_syn.nii.gz"},
      "bucket": "s_real",
      "record_id": "..."
    }

    Output:

    - writes top-level modality keys for LoadImaged:  sample["t1w"], sample["t2w"], ...

    choices structure:
        choices[bucket][modality]["streams"][alias] = stream_key | [stream_key, ...]
        choices[bucket][modality]["probs"][alias] = probability

    Dynamic updates:
    Use SharedChoiceState(shared=True) when using persistent workers.
    Update state.set_choices(...) and state.set_epoch(...) at epoch boundaries.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from monai.transforms import Transform
from omegaconf import ListConfig

logger = logging.getLogger(__name__)



@dataclass
class SharedChoiceState:
    """State shared across DataLoader workers (optional).

    Notes:
      - If `shared=True`, updates made in the main process are visible to
        DataLoader worker processes even with `persistent_workers=True`.
      - Choices are stored as a single dict blob under the manager proxy,
        and replaced wholesale on updates (epoch-boundary).
    """

    choices: Dict[str, Any]
    epoch: int = 0
    seed: int = 0
    shared: bool = False

    def __post_init__(self) -> None:
        if self.shared:
            self._mgr = mp.Manager()
            self._store = self._mgr.dict()
            self._store["choices"] = dict(self.choices)
            self._store["epoch"] = int(self.epoch)
            self._store["seed"] = int(self.seed)
        else:
            self._mgr = None
            self._store = {"choices": dict(self.choices), "epoch": int(self.epoch), "seed": int(self.seed)}

    def __getstate__(self) -> Dict[str, Any]:
        # IMPORTANT:
        # - When shared=True, keep the manager proxy (_store) so workers see updates.
        # - Never try to pickle the Manager object itself (_mgr).
        if self.shared:
            return {
                "shared": True,
                "_store": self._store,
                "choices": {},
                "epoch": 0,
                "seed": 0,
            }

        return {
            "shared": False,
            "choices": self.get_choices(),
            "epoch": self.get_epoch(),
            "seed": self.get_seed(),
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.shared = bool(state.get("shared", False))
        self._mgr = None
        if self.shared:
            self._store = state["_store"]
        else:
            self._store = {
                "choices": dict(state.get("choices", {})),
                "epoch": int(state.get("epoch", 0)),
                "seed": int(state.get("seed", 0)),
            }
        self.choices = self.get_choices()
        self.epoch = self.get_epoch()
        self.seed = self.get_seed()
        
    def get_choices(self) -> Dict[str, Any]:
        return dict(self._store["choices"])

    def set_choices(self, choices: Mapping[str, Any]) -> None:
        logger.info("SharedChoiceState.set_choices called with choices=%s", choices)
        print(f"[ChoiceState] set_choices choices={choices}")
        self._store["choices"] = dict(choices)

    def get_epoch(self) -> int:
        return int(self._store["epoch"])

    def set_epoch(self, epoch: int) -> None:
        logger.info("SharedChoiceState.set_epoch called with epoch=%s", epoch)
        print(f"[ChoiceState] set_epoch epoch={epoch}")
        self._store["epoch"] = int(epoch)

    def get_seed(self) -> int:
        return int(self._store["seed"])

    def set_seed(self, seed: int) -> None:
        logger.info("SharedChoiceState.set_seed called with seed=%s", seed)
        print(f"[ChoiceState] set_seed seed={seed}")
        self._store["seed"] = int(seed)

def _lower_list(values: Sequence[str]) -> list[str]:
    return [str(v).lower() for v in values]

class ChooseStreamForModalitiesd(Transform):
    """Choose one stream per modality and write selections to mapped output keys.

    choices structure:
        choices[bucket][modality]["streams"][alias] = stream_key
        choices[bucket][modality]["probs"][alias] = probability

    You may also provide a wildcard modality "*" under each bucket.

    Note: out_key_map is required and maps stream names (e.g. latent, latent_sigma)
    to output keys written into the batch.

    Choices are stochastic when multiple streams are available.
    """

    def __init__(
        self,
        *,
        modalities: Sequence[str],
        all_modalities: Optional[Sequence[str]] = None,
        is_synthetic_key: Optional[str] = "is_mod_synthetic",
        synthetic_stream_keys: Sequence[str] | str = "synthetic",
        choices: Optional[Mapping[str, Any]] = None,
        state: Optional[SharedChoiceState] = None,
        drop_bucket: bool = True,
        out_key_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.modalities = _lower_list(modalities)

        # We build the is_mod_synthetic vector over *all* modalities.
        # IMPORTANT: internally we always store modalities lowercased so lookups are consistent.
        if all_modalities is None:
            raise ValueError(
                "all_modalities must be provided (length 4), e.g. [T1w, T2w, FLAIR, T1ce] (case-insensitive)."
            )
        if len(all_modalities) != 4:
            raise ValueError(
                f"all_modalities must have length 4 (got {len(all_modalities)}): {list(all_modalities)}"
            )
        self.all_modalities = _lower_list(all_modalities)
        self.drop_bucket = bool(drop_bucket)
        if out_key_map is None:
            raise ValueError("out_key_map must be provided so chosen streams are written to the batch.")
        self.is_synthetic_key = is_synthetic_key
        if isinstance(synthetic_stream_keys, (list, tuple, set, ListConfig)):
            self.synthetic_stream_keys = {str(key) for key in synthetic_stream_keys}
        else:
            self.synthetic_stream_keys = {str(synthetic_stream_keys)}
        self.out_key_map = {str(k): str(v) for k, v in dict(out_key_map or {}).items()}

        if state is not None:
            self.state = state
        else:
            self.state = SharedChoiceState(choices=dict(choices or {}), epoch=0, seed=0, shared=False)

    def _get_choices(self) -> Dict[str, Any]:
        return self.state.get_choices()

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        d = dict(data)

        bucket = str(d.get("bucket", "__default__"))
        choices_table = self._get_choices()

        selected_streams = {}
        synthetic_flags = {}
        stream_key_set = set()
        
        for mod in self.modalities:
            bucket_cfg = choices_table.get(bucket, {}) if isinstance(choices_table, Mapping) else {}
            mod_cfg = {}
            if isinstance(bucket_cfg, Mapping):
                mod_cfg = bucket_cfg.get(mod, bucket_cfg.get("*", {})) or {}

            if not isinstance(mod_cfg, Mapping):
                raise KeyError(f"Missing choice config for bucket='{bucket}', modality='{mod}'.")

            streams_cfg = dict(mod_cfg.get("streams") or {})
            probs_cfg = dict(mod_cfg.get("probs") or {})

            if not streams_cfg:
                raise KeyError(f"Missing streams for bucket='{bucket}', modality='{mod}'.")

            aliases = list(streams_cfg.keys())
            if len(aliases) == 1:
                chosen_alias = aliases[0]
            else:
                weights = []
                for alias in aliases:
                    w = float(probs_cfg.get(alias, 0.0))
                    weights.append(max(0.0, w))
                s = sum(weights)
                if s <= 0:
                    raise ValueError (f"The sum of weights (probabilities) < 0, weights:{weights}, sum weights:{s}")
                else:
                    weights = [w / s for w in weights]
                w = torch.tensor(weights, dtype=torch.float32)
                idx = int(torch.multinomial(w, 1).item())
                chosen_alias = aliases[idx]

            stream_key = streams_cfg.get(chosen_alias)
            if not stream_key:
                raise KeyError(f"Missing stream key for alias '{chosen_alias}' (bucket='{bucket}', modality='{mod}').")
            if isinstance(stream_key, (list, tuple, ListConfig)):
                stream_keys = list(stream_key)
            else:
                stream_keys = [stream_key]
            if not stream_keys:
                raise KeyError(f"Empty stream list for alias '{chosen_alias}' (bucket='{bucket}', modality='{mod}').")
            primary_stream = str(stream_keys[0])
            synthetic_flags[mod] = primary_stream in self.synthetic_stream_keys

            for idx, stream_name in enumerate(stream_keys):
                stream_name = str(stream_name)
                if stream_name not in d:
                    raise KeyError(f"Missing stream '{stream_name}' for modality '{mod}' in bucket '{bucket}'.")
                stream_key_set.add(stream_name)
                mapped_stream_key = stream_name
                if self.out_key_map:
                    mapped_stream_key = self.out_key_map.get(stream_name)
                    if not mapped_stream_key:
                        raise KeyError(f"Missing out_key_map entry for stream '{stream_name}'.")

                stream_dict = d.get(stream_name, None)
                if not isinstance(stream_dict, Mapping):
                    raise KeyError(f"Missing stream '{stream_name}' for modality '{mod}' in bucket '{bucket}'.")
                if mod is None:
                    raise KeyError(
                        f"Missing modality '{mod}' in stream '{stream_name}' for bucket '{bucket}'."
                    )

                value = stream_dict[mod]
                selected_streams.setdefault(mapped_stream_key, {})[mod] = value

        
        for stream_key in stream_key_set:
            d.pop(stream_key, None)

        for stream_key, stream_values in selected_streams.items():
            d[stream_key] = stream_values

        if self.is_synthetic_key:
            d[self.is_synthetic_key] = torch.tensor(
                # synthetic_flags is keyed by lowercased modality names; all_modalities is lowercased too.
                [1 if synthetic_flags.get(mod, False) else 0 for mod in self.all_modalities],
                dtype=torch.long,
            )

        
        if self.drop_bucket and "bucket" in d:
            d.pop("bucket", None)

        return d
