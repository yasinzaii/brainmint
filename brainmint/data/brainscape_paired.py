# brainmint/data/brainscape_paired.py
"""brainmint.data.brainscape_paired

Record-level BrainScape DataModule that emits **multi-modality bundles** ("pairs"/
tuples), bucketizes them by presence/absence rules, and supports **bucket-weighted
sampling** with probabilities that can be updated at **epoch boundaries**.

Design goals
------------
* Keep the existing :class:`brainmint.data.brainscape.BrainScapeDataModule` untouched
  (it is modality-per-sample and useful as-is).
* Reuse the same repo conventions:
  - JSON splits contain records with group dicts like ``preprocessed`` or
    ``synthetic`` mapping ``modality -> relative_path``.
  - Paths are resolved as ``<dataset_root>/<dataset_id>/<subdir>/<relpath>`` by
    default (this matches the current BrainScapeDataModule which hardcodes
    ``subdir='preprocessed'``).
* Make "Stage A/B/C" purely config:
  - eligibility filter via ``pair.require``
  - bucket rules via ``buckets.rules``
  - bucket sampling probabilities via ``set_bucket_probs(...)`` called at epoch
    boundaries (schedule stays outside the DataModule).
* Keep modality names strict (no aliasing): everything is canonical + lowercased.

This module outputs per-sample nested dict streams, e.g.:

    {
      "record_id": "...",
      "dataset": "brats",
      "subject": "sub-001",
      "session": "ses-01",
      "bucket": "s_real",
      "image": {"t1w": "...", "t2w": "...", "flair": "..."},
      "synthetic": {"t2w": "..."}  # optional
      "latent": {...}             # optional
      "seg": {...}                # optional (via seg_map spec)
    }

A downstream transform (see transforms/modality_choice.py) can then choose which
stream to use per modality (real vs synthetic vs latent...) before LoadImaged.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from monai.data import CacheDataset, DataLoader
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


def _as_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, ListConfig)):
        return [str(v) for v in x]
    return [str(x)]


def _load_json_split(json_path: Path, split: str) -> list[dict[str, Any]]:
    with json_path.open("r") as f:
        data = json.load(f)
    if split not in data:
        raise KeyError(f"Split '{split}' not found in JSON: {json_path}")
    if not isinstance(data[split], list):
        raise ValueError(f"JSON split '{split}' must be a list of records")
    return data[split]


@dataclass(frozen=True)
class InputSpec:
    key: str
    group: str
    modalities: tuple[str, ...]
    dataset_root: Path | None = None
    subdir: str = "preprocessed"
    seg_map: bool = False

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> InputSpec:
        key = str(d["key"])
        group = str(d["group"])
        mods = tuple(str(m).lower() for m in _as_list(d.get("modalities")))
        if not mods:
            raise ValueError(f"input_spec '{key}' must define non-empty modalities")
        ds_root = d.get("dataset_root", None)
        ds_root_p = Path(ds_root) if ds_root is not None else None
        subdir = str(d.get("subdir", "preprocessed"))
        seg_map = bool(d.get("seg_map", False))
        return InputSpec(
            key=key,
            group=group,
            modalities=mods,
            dataset_root=ds_root_p,
            subdir=subdir,
            seg_map=seg_map,
        )


@dataclass(frozen=True)
class RequireAny:
    keys: tuple[str, ...]
    modalities: tuple[str, ...]

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> RequireAny:
        keys = tuple(str(k) for k in _as_list(d.get("keys")))
        mods = tuple(str(m).lower() for m in _as_list(d.get("modalities")))
        if not keys or not mods:
            raise ValueError("pair.require.any entries must include non-empty keys and modalities")
        return RequireAny(keys=keys, modalities=mods)


@dataclass(frozen=True)
class BundleConfig:
    emit_keys: tuple[str, ...]
    emit_modalities: tuple[str, ...]
    require_all: dict[str, tuple[str, ...]]
    require_any: tuple[RequireAny, ...]

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, default_modalities: Sequence[str], default_keys: Sequence[str]) -> BundleConfig:
        d = dict(d or {})

        # Required shape:
        # pair:
        #   emit: {keys: [...], modalities: [...]}
        #   require:
        #     all: {image: [...], ...}
        #     any: [{keys:[...], modalities:[...]}, ...]
        if "emit" in d:
            emit = dict(d.get("emit") or {})
            emit_keys = tuple(str(k) for k in _as_list(emit.get("keys", default_keys)))
            emit_mods = tuple(str(m).lower() for m in _as_list(emit.get("modalities", default_modalities)))
        else:
            raise ValueError("pair.emit must be provided with keys and modalities.")

        req = dict(d.get("require") or {})

        all_raw = dict(req.get("all") or {})
        require_all: dict[str, tuple[str, ...]] = {
            str(k): tuple(str(m).lower() for m in _as_list(v))
            for k, v in all_raw.items()
        }

        any_raw = req.get("any", None)
        require_any = tuple(RequireAny.from_dict(x) for x in (any_raw or []))

        if not emit_keys:
            raise ValueError("pair.emit.keys (or pair.keys) must be non-empty")
        if not emit_mods:
            raise ValueError("pair.emit.modalities (or pair.modalities) must be non-empty")

        return BundleConfig(
            emit_keys=emit_keys,
            emit_modalities=emit_mods,
            require_all=require_all,
            require_any=require_any,
        )


@dataclass(frozen=True)
class BucketRule:
    name: str
    includes: dict[str, tuple[str, ...]]
    missing: dict[str, tuple[str, ...]]

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> BucketRule:
        name = str(d["name"])
        inc_raw = dict(d.get("includes") or {})
        mis_raw = dict(d.get("missing") or {})

        def _norm(mm: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
            return {str(k): tuple(str(m).lower() for m in _as_list(v)) for k, v in mm.items()}

        return BucketRule(name=name, includes=_norm(inc_raw), missing=_norm(mis_raw))


def _validate_stream_references(
    *,
    spec_keys: Sequence[str],
    cfg: BundleConfig,
    buckets: Sequence[BucketRule],
    stream_modalities: Mapping[str, Sequence[str]],
) -> None:
    """Catch common config mistakes early.

    We keep modality names strict (no aliasing), and we also keep stream/key
    names strict: any stream referenced by pair.emit/require or buckets must
    correspond to an entry in input_specs.

    This avoids silent bugs where a typo makes a key never populated.
    """
    spec_key_set = set(spec_keys)

    unknown_emit = set(cfg.emit_keys) - spec_key_set
    if unknown_emit:
        raise ValueError(
            f"pair.emit.keys references unknown input_specs key(s): {sorted(unknown_emit)}. "
            f"Known keys: {sorted(spec_key_set)}"
        )

    unknown_req_all = set(cfg.require_all.keys()) - set(cfg.emit_keys)
    if unknown_req_all:
        raise ValueError(
            f"pair.require.all references key(s) not in pair.emit.keys: {sorted(unknown_req_all)}"
        )

    for ra in cfg.require_any:
        unknown = set(ra.keys) - set(cfg.emit_keys)
        if unknown:
            raise ValueError(
                f"pair.require.any references key(s) not in pair.emit.keys: {sorted(unknown)}"
            )

    def _validate_mods(stream: str, mods: Sequence[str], context: str) -> None:
        allowed = {str(m).lower() for m in stream_modalities.get(stream, [])}
        unknown_mods = {str(m).lower() for m in mods} - allowed
        if unknown_mods:
            raise ValueError(
                f"{context} references modality(s) {sorted(unknown_mods)} for stream '{stream}'. "
                f"Allowed modalities: {sorted(allowed)}"
            )

    for stream, mods in cfg.require_all.items():
        _validate_mods(stream, mods, "pair.require.all")

    for ra in cfg.require_any:
        for m in ra.modalities:
            matches = [
                k
                for k in ra.keys
                if str(m).lower() in {str(mm).lower() for mm in stream_modalities.get(k, [])}
            ]
            if not matches:
                raise ValueError(
                    f"pair.require.any references modality '{m}' but none of keys {list(ra.keys)} "
                    "can emit that modality."
                )

    # Bucket rules must reference emitted stream keys (otherwise they can never match).
    emit_key_set = set(cfg.emit_keys)
    for br in buckets:
        for k in list(br.includes.keys()) + list(br.missing.keys()):
            if k not in spec_key_set:
                raise ValueError(
                    f"Bucket '{br.name}' references unknown stream '{k}'. Known keys: {sorted(spec_key_set)}"
                )
            if k not in emit_key_set:
                raise ValueError(
                    f"Bucket '{br.name}' references stream '{k}' that is not emitted. "
                    f"Add it to pair.emit.keys."
                )
        for stream, mods in br.includes.items():
            _validate_mods(stream, mods, f"Bucket '{br.name}' includes")
        for stream, mods in br.missing.items():
            _validate_mods(stream, mods, f"Bucket '{br.name}' missing")


def _resolve_path(
    *,
    dataset_id: str,
    rel: str,
    spec: InputSpec,
    default_root: Path,
) -> Path:
    # If rel is absolute, respect it.
    p = Path(rel)
    if p.is_absolute():
        return p
    root = spec.dataset_root or default_root
    return root / dataset_id / spec.subdir / rel


def _make_record_id(rec: Mapping[str, Any]) -> str:
    # stable key for deterministic per-record decisions
    ds = str(rec.get("dataset", ""))
    sub = str(rec.get("subject", ""))
    ses = str(rec.get("session", ""))
    rtype = str(rec.get("type", ""))
    group = str(rec.get("group", ""))
    return "|".join([ds, sub, ses, rtype, group])


def _has_stream_mod(sample: Mapping[str, Any], stream: str, mod: str) -> bool:
    sd = sample.get(stream, None)
    if not isinstance(sd, Mapping):
        return False
    return str(mod).lower() in {str(k).lower() for k in sd.keys()}


def _bucket_matches(sample: Mapping[str, Any], rule: BucketRule) -> bool:
    for stream, mods in rule.includes.items():
        for m in mods:
            if not _has_stream_mod(sample, stream, m):
                return False
    for stream, mods in rule.missing.items():
        for m in mods:
            if _has_stream_mod(sample, stream, m):
                return False
    return True


def _passes_require_rules(sample: Mapping[str, Any], cfg: BundleConfig) -> bool:
    # require_all: for each key -> all modalities must exist under that stream
    for stream, mods in cfg.require_all.items():
        for m in mods:
            if not _has_stream_mod(sample, stream, m):
                return False

    # require_any: each item requires each modality exists in at least one of its keys
    for ra in cfg.require_any:
        for m in ra.modalities:
            ok = any(_has_stream_mod(sample, k, m) for k in ra.keys)
            if not ok:
                return False

    return True


def _build_record_datalist(
    *,
    records: Sequence[Mapping[str, Any]],
    input_specs: Sequence[InputSpec],
    cfg: BundleConfig,
    default_root: Path,
    verify_paths: bool,
    bucket_rules: Sequence[BucketRule],
    exhaustive_buckets: bool,
    exclusive_buckets: bool,
    default_meta_keys: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    items: list[dict[str, Any]] = []

    bucket_to_indices: dict[str, list[int]] = {b.name: [] for b in bucket_rules}
    meta_keys = [str(k) for k in (default_meta_keys or [])]

    for rec in records:
        dataset_id = str(rec.get("dataset", ""))
        if not dataset_id:
            raise ValueError("Record missing required field: 'dataset'")
        subject = str(rec.get("subject", ""))
        session = str(rec.get("session", ""))
        demographics = rec.get("demographics", {})

        sample: dict[str, Any] = {
            "record_id": _make_record_id(rec),
            "dataset": dataset_id,
            "subject": subject,
            "session": session,
            "demographics": demographics,
        }

        for k in meta_keys:
            if k in sample and k != "modality":
                raise ValueError(f"meta_keys should not include reserved sample key: {k}")
            if k == "modality":
                continue
            if k in rec:
                sample[k] = rec.get(k)

        # Build stream dicts from specs (only for emitted keys + emitted modalities).
        for spec in input_specs:
            if spec.key not in cfg.emit_keys:
                continue

            group = rec.get(spec.group, None)
            if not group:
                continue

            if not isinstance(group, Mapping):
                raise ValueError(f"Record group '{spec.group}' must be a mapping.")

            group_lc = {str(k).lower(): str(v) for k, v in group.items()}

            allowed_mods = set(cfg.emit_modalities)
            if spec.seg_map:
                # seg_map can emit modalities not in cfg.emit_modalities (e.g. "seg").
                allowed_mods |= set(spec.modalities)

            out_stream: dict[str, str] = {}
            for m in spec.modalities:
                m_lc = str(m).lower()
                if m_lc not in allowed_mods:
                    continue

                rel = group_lc.get(m_lc, None)
                if rel is None:
                    continue
                    # raise ValueError(
                    #     f"Record group '{spec.group}' missing expected modality key: {m_lc}"
                    # )

                fpath = _resolve_path(dataset_id=dataset_id, rel=rel, spec=spec, default_root=default_root)
                if verify_paths and not fpath.is_file():
                    raise FileNotFoundError(
                        f"Missing file: dataset={dataset_id} subject={subject} "
                        f"stream={spec.key} group={spec.group} modality={m_lc} path={fpath}"
                    )
                out_stream[m_lc] = str(fpath)

            if out_stream:
                sample[spec.key] = out_stream

        # TODO! Check if we need the following logic. Raising Error For now, modality should not be in metadata
        # if "modality" in meta_keys and "modality" not in sample:
        #     present_mods = set()
        #     for stream in cfg.emit_keys:
        #         stream_dict = sample.get(stream, {})
        #         if isinstance(stream_dict, Mapping):
        #             present_mods.update({str(k).lower() for k in stream_dict.keys()})
        #     sample["modality"] = sorted(present_mods)
        if "modality" in meta_keys:
            raise KeyError("Modality key is not supported for Paird BrainScape Samples!")


        # Eligibility filter
        if not _passes_require_rules(sample, cfg):
            continue

        # If no bucket rules are provided, still attach a stable default bucket name.
        # (Does NOT enable bucket sampling; it just makes downstream code/configs simpler.)
        if not bucket_rules:
            sample.setdefault("bucket", "__default__")
            if "__default__" not in bucket_to_indices:
                bucket_to_indices["__default__"] = []
        
        
        # Bucket assignment
        if bucket_rules:
            hits = [b.name for b in bucket_rules if _bucket_matches(sample, b)]
            if exclusive_buckets and len(hits) > 1:
                raise RuntimeError(
                    f"Record {sample['record_id']} matches multiple buckets {hits}. "
                    "Fix buckets.rules to be disjoint."
                )
            if exhaustive_buckets and len(hits) == 0:
                raise RuntimeError(
                    f"Record {sample['record_id']} did not match any bucket. "
                    "Fix buckets.rules or add a catch-all bucket."
                )
            if hits:
                sample["bucket"] = hits[0]

        items.append(sample)
        if "bucket" in sample:
            if bucket_rules or sample["bucket"] == "__default__":
                bucket_to_indices[sample["bucket"]].append(len(items) - 1)

    # Final exhaustive sanity: if buckets exist + exhaustive, then every item must have bucket.
    if bucket_rules and exhaustive_buckets:
        for it in items:
            if "bucket" not in it:
                raise RuntimeError(
                    f"Internal error: item passed exhaustive bucketization without bucket: {it.get('record_id')}"
                )

    return items, bucket_to_indices


def _make_bucket_sample_weights(
    *,
    n_items: int,
    bucket_to_indices: Mapping[str, Sequence[int]],
    bucket_probs: Mapping[str, float],
) -> torch.Tensor:
    weights = torch.zeros(n_items, dtype=torch.double)

    for bname, idxs in bucket_to_indices.items():
        p = float(bucket_probs.get(bname, 0.0))
        if p < 0:
            raise ValueError("bucket_probs must be non-negative")
        if len(idxs) == 0:
            continue
        # Each item in bucket gets equal weight; bucket mass is p.
        w = p / float(len(idxs))
        for i in idxs:
            weights[int(i)] = w

    if float(weights.sum().item()) <= 0:
        raise ValueError("bucket_probs sum over non-empty buckets must be > 0")

    return weights


def _plain_dm_hparams(**kw):
    out = {}
    for k, v in kw.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif OmegaConf.is_config(v):
            out[k] = OmegaConf.to_container(v, resolve=True)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _to_jsonish(x):
    # OmegaConf / DictConfig / ListConfig -> plain
    if OmegaConf.is_config(x):
        return OmegaConf.to_container(x, resolve=True, throw_on_missing=False)

    # Paths -> str
    if isinstance(x, Path):
        return str(x)

    # dict-like
    if isinstance(x, dict):
        return {str(k): _to_jsonish(v) for k, v in x.items()}

    # list/tuple/set -> list
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonish(v) for v in x]

    # basic scalars
    return x


class BrainScapePairedDataModule(pl.LightningDataModule):
    """Record-level multi-stream BrainScape DataModule.

    Notes on probability updates
    ----------------------------
    Call `set_bucket_probs({...})` at *epoch boundaries* (e.g. `on_train_epoch_start`)
    to change the train sampler distribution. Do not change it mid-epoch.
    """

    def __init__(
        self,
        json_path: str | Path,
        dataset_root: str | Path,
        *,
        modalities: Sequence[str],
        input_specs: Sequence[Mapping[str, Any]],
        pair: Mapping[str, Any],
        buckets: Mapping[str, Any] | None = None,
        bucket_probs: Mapping[str, float] | None = None,
        use_bucket_sampler: bool = True,
        verify_paths: bool = True,
        # transforms
        train_tf=None,
        val_tf=None,
        test_tf=None,
        # dataloader knobs
        batch_size: int = 1,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        cache_rate: float = 0.0,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        subset_frac: float | None = None,
        seed: int = 42,
        dataset_filter: Sequence[str] | None = None,
        dataset_filter_exclude: Sequence[str] | None = None,
        # compatibility knob (present in your configs; not required here)
        default_meta_keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__()

        hp = dict(
            json_path=json_path,
            dataset_root=dataset_root,
            modalities=modalities,
            input_specs=input_specs,
            pair=pair,
            buckets=buckets,
            bucket_probs=bucket_probs,
            use_bucket_sampler=use_bucket_sampler,
            verify_paths=verify_paths,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            cache_rate=cache_rate,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            subset_frac=subset_frac,
            seed=seed,
            dataset_filter=dataset_filter,
            dataset_filter_exclude=dataset_filter_exclude,
            default_meta_keys=default_meta_keys,
        )
        # HP's ignored: ["train_tf", "val_tf", "test_tf"]
        self.save_hyperparameters(_to_jsonish(hp), )


        self.json_path = Path(json_path)
        self.dataset_root = Path(dataset_root)

        self.modalities = [str(m).lower() for m in modalities]  # canonical order, strict names
        self.input_specs = [InputSpec.from_dict(d) for d in input_specs]

        spec_keys = [s.key for s in self.input_specs]
        self.bundle_cfg = BundleConfig.from_dict(
            pair,
            default_modalities=self.modalities,
            default_keys=spec_keys,
        )

        # Strict modality names: pair.emit.modalities must be subset of module modalities
        unknown_mods = set(self.bundle_cfg.emit_modalities) - set(self.modalities)
        if unknown_mods:
            raise ValueError(
                f"pair.emit.modalities contains unknown modalities {sorted(unknown_mods)}. "
                f"Allowed (DataModule modalities): {self.modalities}"
            )

        # input_specs modalities must be strict too (unless seg_map)
        for s in self.input_specs:
            if s.seg_map:
                continue
            extra = set(s.modalities) - set(self.modalities)
            if extra:
                raise ValueError(
                    f"input_spec '{s.key}' contains modalities not in DataModule modalities: {sorted(extra)}"
                )

        # Parse buckets
        buckets = dict(buckets or {})
        rules_raw = buckets.get("rules", None) or []
        self.bucket_rules: list[BucketRule] = [BucketRule.from_dict(r) for r in rules_raw]
        self.exhaustive_buckets: bool = bool(buckets.get("exhaustive", True))
        self.exclusive_buckets: bool = bool(buckets.get("exclusive", True))

        # Ensure unique bucket names.
        bnames = [b.name for b in self.bucket_rules]
        if len(set(bnames)) != len(bnames):
            raise ValueError(f"Duplicate bucket names in buckets.rules: {bnames}")

        # Build stream modalities for validation (emit + seg_map allowances).
        stream_modalities: dict[str, list[str]] = {k: [] for k in spec_keys}
        emit_mods = set(self.bundle_cfg.emit_modalities)
        for spec in self.input_specs:
            allowed = set(spec.modalities) if spec.seg_map else (set(spec.modalities) & emit_mods)
            stream_modalities.setdefault(spec.key, [])
            stream_modalities[spec.key].extend(sorted(allowed))

        for k in self.bundle_cfg.emit_keys:
            if not stream_modalities.get(k):
                raise ValueError(
                    f"pair.emit.keys includes '{k}' but no modalities are available to emit for that stream. "
                    "Check input_specs and pair.emit.modalities."
                )

        # Validate stream references (emit/require + bucket rule keys)
        _validate_stream_references(
            spec_keys=spec_keys,
            cfg=self.bundle_cfg,
            buckets=self.bucket_rules,
            stream_modalities=stream_modalities,
        )

        # bucket probs default: proportional to bucket size (uniform over all samples)
        self._bucket_probs_user: dict[str, float] = dict(bucket_probs or {})
        self.use_bucket_sampler = bool(use_bucket_sampler)
        self.verify_paths = bool(verify_paths)

        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size) if val_batch_size is not None else int(batch_size)
        self.test_batch_size = int(test_batch_size) if test_batch_size is not None else int(batch_size)

        self.cache_rate = float(cache_rate)
        if self.cache_rate != 0.0:
            # CacheDataset would freeze dynamic stream choices and sampling updates between epochs.
            raise ValueError("cache_rate must be 0.0 for dynamic stream selection/sampling.")
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers) if self.num_workers > 0 else False
        self.pin_memory = bool(pin_memory)
        self.subset_frac = subset_frac
        self.seed = int(seed)

        # Optional dataset include/exclude filters (applied when reading splits).
        # - dataset_filter: keep only these dataset IDs
        # - dataset_filter_exclude: drop these dataset IDs
        self.dataset_filter = [str(x).lower() for x in (dataset_filter or [])]
        self.dataset_filter_exclude = [str(x).lower() for x in (dataset_filter_exclude or [])]

        self.train_tf = train_tf
        self.val_tf = val_tf
        self.test_tf = test_tf

        self.default_meta_keys = [str(k) for k in (default_meta_keys or [])]

        self._datasets: dict[str, CacheDataset] = {}
        self._bucket_to_indices: dict[str, dict[str, list[int]]] = {}
        self._train_sampler: WeightedRandomSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        # build datasets lazily; stage can be "fit"/"test" etc.
        if stage in (None, "fit", "train"):
            train_records = _load_json_split(self.json_path, "train")
            train_records = self._filter_records_by_dataset(train_records)
            val_records = _load_json_split(self.json_path, "val")
            val_records = self._filter_records_by_dataset(val_records)
            if self.subset_frac is not None:
                train_records = self._subset_records(train_records, self.subset_frac)
                val_records = self._subset_records(val_records, self.subset_frac)

            train_items, train_buckets = _build_record_datalist(
                records=train_records,
                input_specs=self.input_specs,
                cfg=self.bundle_cfg,
                default_root=self.dataset_root,
                verify_paths=self.verify_paths,
                bucket_rules=self.bucket_rules,
                exhaustive_buckets=self.exhaustive_buckets,
                exclusive_buckets=self.exclusive_buckets,
                default_meta_keys=self.default_meta_keys,
            )
            val_items, val_buckets = _build_record_datalist(
                records=val_records,
                input_specs=self.input_specs,
                cfg=self.bundle_cfg,
                default_root=self.dataset_root,
                verify_paths=self.verify_paths,
                bucket_rules=self.bucket_rules,
                exhaustive_buckets=self.exhaustive_buckets,
                exclusive_buckets=self.exclusive_buckets,
                default_meta_keys=self.default_meta_keys,
            )

            self._datasets["train"] = CacheDataset(train_items, transform=self.train_tf, cache_rate=self.cache_rate)
            self._datasets["val"] = CacheDataset(val_items, transform=self.val_tf, cache_rate=self.cache_rate)
            self._bucket_to_indices["train"] = train_buckets
            self._bucket_to_indices["val"] = val_buckets
            self._print_bucket_stats(split_name="train", bucket_to_indices=train_buckets)
            self._print_bucket_stats(split_name="val", bucket_to_indices=val_buckets)

            self._build_train_sampler()

        if stage in ("validate",):
            val_records = _load_json_split(self.json_path, "val")
            val_records = self._filter_records_by_dataset(val_records)
            if self.subset_frac is not None:
                val_records = self._subset_records(val_records, self.subset_frac)
            val_items, val_buckets = _build_record_datalist(
                records=val_records,
                input_specs=self.input_specs,
                cfg=self.bundle_cfg,
                default_root=self.dataset_root,
                verify_paths=self.verify_paths,
                bucket_rules=self.bucket_rules,
                exhaustive_buckets=self.exhaustive_buckets,
                exclusive_buckets=self.exclusive_buckets,
                default_meta_keys=self.default_meta_keys,
            )
            self._datasets["val"] = CacheDataset(val_items, transform=self.val_tf, cache_rate=self.cache_rate)
            self._bucket_to_indices["val"] = val_buckets
            self._print_bucket_stats(split_name="val", bucket_to_indices=val_buckets)

        if stage in (None, "test"):
            test_records = _load_json_split(self.json_path, "test")
            test_records = self._filter_records_by_dataset(test_records)
            if self.subset_frac is not None:
                test_records = self._subset_records(test_records, self.subset_frac)
            test_items, test_buckets = _build_record_datalist(
                records=test_records,
                input_specs=self.input_specs,
                cfg=self.bundle_cfg,
                default_root=self.dataset_root,
                verify_paths=self.verify_paths,
                bucket_rules=self.bucket_rules,
                exhaustive_buckets=self.exhaustive_buckets,
                exclusive_buckets=self.exclusive_buckets,
                default_meta_keys=self.default_meta_keys,
            )
            self._datasets["test"] = CacheDataset(test_items, transform=self.test_tf, cache_rate=self.cache_rate)
            self._bucket_to_indices["test"] = test_buckets
            self._print_bucket_stats(split_name="test", bucket_to_indices=test_buckets)

    def _filter_records_by_dataset(self, records):
        """Apply dataset_filter / dataset_filter_exclude to record list.

        Filters are case-insensitive and match on record['dataset'].
        """
        if not self.dataset_filter and not self.dataset_filter_exclude:
            return list(records)

        include = set(self.dataset_filter) if self.dataset_filter else None
        exclude = set(self.dataset_filter_exclude) if self.dataset_filter_exclude else None

        out = []
        for r in records:
            ds = str(r.get("dataset", "")).lower()
            if not ds:
                continue
            if include is not None and ds not in include:
                continue
            if exclude is not None and ds in exclude:
                continue
            out.append(r)
        return out


    def _subset_records(self, records: Sequence[Mapping[str, Any]], frac: float) -> list[Mapping[str, Any]]:
        if not (0.0 < float(frac) <= 1.0):
            raise ValueError("subset_frac must be in (0, 1]")
        n = max(1, int(round(len(records) * float(frac))))
        rng = random.Random(self.seed)
        idxs = rng.sample(range(len(records)), k=min(n, len(records)))
        return [records[i] for i in idxs]

    def _build_train_sampler(self) -> None:
        if not self.bucket_rules or not self.use_bucket_sampler:
            self._train_sampler = None
            return

        ds = self._datasets.get("train", None)
        if ds is None:
            return
        if len(ds) == 0:
            self._train_sampler = None
            return

        buckets = self._bucket_to_indices["train"]
        # default probs: proportional to bucket size (so uniform overall)
        probs: dict[str, float] = {}
        total = sum(len(v) for v in buckets.values())
        for b, idxs in buckets.items():
            probs[b] = (len(idxs) / total) if total > 0 else 0.0

        # override with user-provided
        for b, p in self._bucket_probs_user.items():
            probs[str(b)] = float(p)

        # Validate bucket names
        unknown = set(probs.keys()) - set(buckets.keys())
        if unknown:
            raise ValueError(f"bucket_probs contains unknown bucket(s): {sorted(unknown)}")

        weights = _make_bucket_sample_weights(
            n_items=len(ds),
            bucket_to_indices=buckets,
            bucket_probs=probs,
        )
        self._train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)

    def set_bucket_probs(self, probs: Mapping[str, float]) -> None:
        """Update train bucket sampling probabilities (epoch-boundary safe)."""
        if not self.bucket_rules:
            raise RuntimeError("No buckets configured; cannot set bucket probabilities.")

        probs = {str(k): float(v) for k, v in dict(probs).items()}
        buckets = self._bucket_to_indices.get("train", None)
        if buckets is None:
            raise RuntimeError("DataModule not set up yet; call setup('fit') before set_bucket_probs().")

        unknown = set(probs.keys()) - set(buckets.keys())
        if unknown:
            raise ValueError(f"set_bucket_probs: unknown bucket(s): {sorted(unknown)}")

        # Fill missing buckets with 0 by default
        full = {b: float(probs.get(b, 0.0)) for b in buckets.keys()}
        self._bucket_probs_user = dict(full)
        weights = _make_bucket_sample_weights(n_items=len(self._datasets["train"]), bucket_to_indices=buckets, bucket_probs=full)

        if self._train_sampler is None:
            # build if needed
            self._train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(self._datasets["train"]), replacement=True)
        else:
            # update in-place
            self._train_sampler.weights = weights
        logger.info("set_bucket_probs updated bucket_probs=%s", full)
        print(f"[BucketState] set_bucket_probs bucket_probs={full}")

        
    def _print_bucket_stats(self, split_name: str, bucket_to_indices: dict[str, list[int]]) -> None:
        # Total samples counted by bucket assignment
        total = sum(len(v) for v in bucket_to_indices.values())

        # Sort buckets by count (descending) for readability
        sorted_items = sorted(bucket_to_indices.items(), key=lambda kv: len(kv[1]), reverse=True)

        lines = []
        lines.append(f"[BrainScapePairedDataModule] Bucket stats | split={split_name} | total={total}")
        for bname, idxs in sorted_items:
            n = len(idxs)
            pct = (100.0 * n / total) if total > 0 else 0.0
            lines.append(f"  - {bname:20s} n={n:6d} ({pct:6.2f}%)")
        msg = "\n".join(lines)

        # Print to console
        print(msg)
        logger.info(msg)
        
    def get_bucket_stats(self, split: str = "train") -> dict[str, int]:
        b = self._bucket_to_indices.get(split, {})
        return {k: len(v) for k, v in b.items()}

    def train_dataloader(self):
        ds = self._datasets["train"]
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=(self._train_sampler is None),
            sampler=self._train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            generator=generator,
        )

    def val_dataloader(self):
        ds = self._datasets["val"]
        return DataLoader(
            ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        ds = self._datasets["test"]
        return DataLoader(
            ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
