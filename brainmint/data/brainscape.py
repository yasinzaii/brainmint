# BrainScapeDataModule: Builds and returns train/val/test DataLoaders for the BrainScape MRI dataset.

import json
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import pytorch_lightning as pl

from monai.data import CacheDataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from monai.transforms import Compose

_LOG = logging.getLogger(__name__)

def _load_json_split(json_path: Path | str, split: str) -> List[Dict[str, Any]]:
    """Return list[dict] for the requested split (train|val|test) name."""
    json_path = Path(json_path)
    with json_path.open("r") as fp:
        data = json.load(fp)
    if split not in data:
        raise KeyError(f"Split '{split}' not found in {json_path}")
    return data[split]



def _build_datalist(
    entries: List[Dict[str, Any]],
    specs: Tuple[Dict[str, Any], ...],
    modalities: Tuple[str, ...],
    dataset_filter: Optional[Tuple[str, ...]] = None,
) -> List[Dict[str, Any]]:
    """Flatten JSON records into sample dictionaries.

    Each spec describes one tensor to load, with:
      - key: output key in sample dict
      - group: JSON dict field containing modality->relative_path mapping
      - dataset_root (optional): filesystem root for this spec

    NOTE: spec["dataset_root"] now fully controls where paths are resolved.
    If dataset_filter is set, only records whose dataset ID matches are used.
    """

    items: List[Dict[str, Any]] = []
    if not modalities:
        raise RuntimeError("'modalities' must be provided")

    mods = [m.lower() for m in modalities]
    dataset_filter_set = {d.lower() for d in dataset_filter} if dataset_filter else None

    for rec in entries:
        dataset_id = rec["dataset"]
        if dataset_filter_set and dataset_id.lower() not in dataset_filter_set:
            continue

        for mod in mods:
            sample: Dict[str, Any] = {
                "dataset": dataset_id,
                "modality": mod,
                "subject": rec.get("subject"),
                "demographics": rec.get("demographics", {}),
            }

            # Optional metadata used by downstream match keys (e.g., synthetic generation)
            for opt_key in ("session", "type", "group"):
                if opt_key in rec:
                    sample[opt_key] = rec.get(opt_key)
            
            # TODO: Transform Module cannot handle Image+Label with missing Label
            
            spec_item_added = False
            for spec in specs:
                target_mod = mod  # Modality that will be appended to sample

                spec_mods = spec.get("modalities")
                if isinstance(spec_mods, str):
                    spec_mods = [spec_mods]
                spec_mods = [m.lower() for m in spec_mods]

                if spec_mods and mod not in spec_mods:
                    # Only include SEG MAPs with other modalities
                    is_seg_map = bool(spec.get("seg_map", False))
                    if is_seg_map:
                        if len(spec_mods) > 1:
                            raise ValueError(f"BrainScape Module Currently only supports 1 SEG MASK In a SPEC but got {len(spec_mods)}, modalities:{spec_mods}")
                        target_mod = spec_mods[0]
                    else:
                        continue  # Don't proceed if modality is not required [not in the spec]


                group_dict = rec.get(spec["group"], None)
                if not isinstance(group_dict, dict) or group_dict is None:
                    #raise ValueError(f"Missing Group:{spec['group']} from Rec Sample: {rec}")
                    continue

                gdict = {k.lower(): v for k, v in group_dict.items()}
                rel = gdict.get(target_mod)
                if rel is None:
                    #_LOG.info(f"Missing '{spec['key']}' for modality '{mod}' in dataset '{dataset_id}'")
                    continue

                # Resolve root per spec
                spec_root = spec.get("dataset_root", None)
                if spec_root is None:
                    raise KeyError(f"Spec for key='{spec.get('key')}' must define 'dataset_root' ")
                root = Path(spec_root)

                # All groups stored under <root>/<dataset_id>/preprocessed/<rel>
                fpath = root / dataset_id / "preprocessed" / rel
                if not fpath.is_file():
                    raise FileNotFoundError(
                        f"Referenced file for key '{spec['key']}' (modality '{target_mod}', "
                        f"group '{spec['group']}', dataset '{dataset_id}') does not exist: {fpath}"
                    )

                spec_item_added = True
                sample[spec["key"]] = str(fpath)

            # Appends after all of the Specs have been added!
            # Inputs, Latent and Masks!
            if spec_item_added:
                items.append(sample)

    return items

    
class BrainScapeDataModule(pl.LightningDataModule):
    """Builds and returns MONAI CacheDatasets + DataLoaders for BrainScape MRI."""

    def __init__(
        self,
        json_path: str | Path,
        dataset_root: str | Path,
        train_tf: Optional[Compose] = None,
        val_tf: Optional[Compose] = None,
        test_tf: Optional[Compose] = None,
        batch_size: int = 1,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        cache_rate: float = 1,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        subset_frac: Optional[float] = None,
        seed: int = 42,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        input_specs: Optional[List[Dict[str, Any]]] = None,
        modalities: List[str] | Tuple[str, ...] = ("t1w", "t2w", "t1ce", "flair"), 
        use_modality_sampler: bool = False, # WeightedRandomSampler
        modality_sampler_alpha: float = 0.7,
        dataset_filter: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.json_path          = Path(json_path)
        self.dataset_root       = Path(dataset_root)

        self.cache_rate         = cache_rate
        self.num_workers        = num_workers
        self.prefetch_factor    = prefetch_factor
        self.subset_frac        = subset_frac # Generate sets with Frac of MRI's for testing
        self.seed               = seed
        self.pin_memory         = pin_memory
        self.persistent_workers = persistent_workers

        default_spec = {
            "key": "image", 
            "group": "preprocessed", 
            "modalities": {"t1w", "t2w", "t1ce", "flair"},
            "dataset_root": self.dataset_root, # Optional
        }

        self.input_specs  = tuple(input_specs) if input_specs is not None else (default_spec,)
        for spec in self.input_specs:
            if "dataset_root" not in spec.keys():
                spec.update({"dataset_root": self.dataset_root})

        self.modalities   = tuple(m.lower() for m in modalities)
        self.split_names    = ("train", "val", "test")
        
        self.use_modality_sampler = bool(use_modality_sampler)
        self.modality_sampler_alpha = float(modality_sampler_alpha)

        self._tf = {
            "train": train_tf,
            "val": val_tf,
            "test": test_tf,
        }

        self._bs = {
            "train":batch_size,
            "val":val_batch_size or batch_size,
            "test":test_batch_size or batch_size,
        }

        self._datasets: Dict[str, CacheDataset] = {}
        self.generator = torch.Generator().manual_seed(seed)
        self.dataset_filter = tuple(str(d) for d in dataset_filter) if dataset_filter else None


    def setup(self, stage: str | None = None) -> None:
        """Build datasets and transforms once."""
        if self.dataset_filter:
            _LOG.warning("Dataset filter enabled: %s", ", ".join(self.dataset_filter))

        requested_splits: list[str] = []

        # Lightning semantics:
        # - "fit": train + val
        # - "validate": val
        # - "test": test
        if stage in (None, "fit", "train"):
            requested_splits.extend(["train", "val"])

        if stage in (None, "validate"):
            requested_splits.append("val")

        if stage in (None, "test"):
            requested_splits.append("test")

        # Deduplicate while preserving order
        requested_splits = list(dict.fromkeys(requested_splits))

        # Fallback if some other stage (e.g. "predict") calls setup
        if not requested_splits:
            requested_splits = list(getattr(self, "target_splits", ["train", "val", "test"]))

        # Only build splits that are not yet present
        to_build = [s for s in requested_splits if s not in self._datasets]
        if not to_build:
            return

        # Load JSON splits just for needed splits
        splits = {
            sp: _load_json_split(self.json_path, sp)
            for sp in to_build
        }

        # Optional sub‑sampling for quick debug runs (for testing)
        if self.subset_frac:
            self.subset_frac = float(self.subset_frac)
            if 0.0 < self.subset_frac < 1.0:
                random.seed(self.seed)
                for k, v in splits.items():
                    n_keep = max(1, int(len(v) * self.subset_frac))
                    splits[k] = random.sample(v, n_keep)

        # Build datalists
        lists = {
            k: _build_datalist(
                entries = v,
                specs=self.input_specs,
                modalities=self.modalities,
                dataset_filter=self.dataset_filter,
            )
            for k, v in splits.items()
        }

        # Check for empty data lists (only for splits we actually care about)
        for split in to_build:
            datalist = lists.get(split)
            if not datalist:
                raise RuntimeError(f"No samples found for split '{split}'")

        # Build CacheDatasets
        for split_name in to_build:
            self._datasets[split_name] = CacheDataset(
                data=lists[split_name],
                transform=self._tf[split_name],
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )

    def _make_loader(
        self,
        split: str,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> DataLoader:
        """Build & Return DataLoader."""
        return DataLoader(
            dataset=self._datasets[split],
            batch_size=self._bs[split],
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            generator=self.generator,
        )

    def _ensure_built(self):
        if not self._datasets:
            self.setup()
    
    
    def get_dataset(self, name: str = "train") -> CacheDataset:
        self._ensure_built()
        if name not in self._datasets:
            raise KeyError(f"Dataset split '{name}' is not available. Available splits: {list(self._datasets.keys())}")     
        return self._datasets[name]

    def train_dataloader(self):
        self._ensure_built()
        if self.use_modality_sampler:
            # Use your weighted sampler-based loader
            return self.weighted_train_dataloader(alpha=self.modality_sampler_alpha)
        
        # Default behaviour (no balancing)
        return self._make_loader("train", shuffle=True)

    def val_dataloader(self):
        self._ensure_built()
        return self._make_loader("val",   shuffle=False)

    def test_dataloader(self):
        self._ensure_built()
        return self._make_loader("test",  shuffle=False)

    
    def get_splits_length(self) -> Dict[str, int]:
        """Get the total number of samples for all dataset splits."""
        return {split: len(dataset) for split, dataset in self._datasets.items()}


    def _build_modality_sampler(
        self,
        split: str = "train",
        alpha: float = 0.7,
    ) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler for the given split based on modality frequencies.
        """
        if split not in self._datasets:
            raise RuntimeError(f"Dataset split '{split}' has not been built yet")

        ds = self._datasets[split]

        # Collect modality labels for every sample (as strings)
        labels_str: list[str] = []
        for item in ds.data:
            mod = item["modality"]
            labels_str.append(str(mod).lower())

        # Map modalities to integer indices using the module's modality list
        # This keeps the ordering consistent with self.modalities
        mod_to_idx = {m: idx for idx, m in enumerate(self.modalities)}

        # Convert string labels to integer indices
        label_indices = []
        for m in labels_str:
            if m not in mod_to_idx:
                raise ValueError(f"Unknown modality {m!r} encountered in dataset")
            label_indices.append(mod_to_idx[m])

        label_indices = torch.tensor(label_indices, dtype=torch.long)

        num_classes = len(self.modalities)
        class_counts = torch.bincount(label_indices, minlength=num_classes).float()

        # Avoid division-by-zero just in case
        class_counts[class_counts == 0] = 1.0

        # Alpha-tempered inverse-frequency weights
        # w_c = (1 / count_c) ** alpha
        class_weights = (1.0 / class_counts) ** float(alpha)

        # Per-sample weights: w_i = w_{class(label_i)}
        sample_weights = class_weights[label_indices]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(ds),
            replacement=True,
        )
        return sampler

    def weighted_train_dataloader(self, alpha: float = 0.7) -> DataLoader:
        """
        Return a train DataLoader that uses a WeightedRandomSampler to
        rebalance modalities according to an alpha-tempered scheme.
        """
        self._ensure_built()
        sampler = self._build_modality_sampler(split="train", alpha=alpha)

        return DataLoader(
            dataset=self._datasets["train"],
            batch_size=self._bs["train"],
            sampler=sampler,
            shuffle=False,  # IMPORTANT: must be False when sampler is used
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            generator=self.generator,
        )
