
from collections.abc import Mapping
from typing import Any

import torch
from monai.transforms import MapTransform


class DemographicsConditioningd(MapTransform):
    """
    Prepare BrainScape demographics for model conditioning.

    Output (per sample):
        data[out_dict_key]    # "demo_dict"
            {field_name: parsed_value}
            - numeric fields: normalized float (z-score)
            - categorical/YND: integer index (0 .. K-1)

        data[out_values_key]  # "demo_values"
            1D float32 tensor of shape (F,)
            - one slot per field in config["ordered_fields"]
            - numeric: normalized float
            - categorical/YND: float(index)

        data[out_missing_key] # "demo_missing"
            1D bool tensor of shape (F,)
            - True if missing or explicitly "n/a"
            - False otherwise

    Notes:
    - For numeric fields, missing or "n/a" → value=0.0 (which corresponds
      to mean after normalization) + missing=True.
    - For categorical/YND fields, missing or "n/a" → index given by
      mapping["n/a"] (typically 0) + missing=True.
    - Any unexpected string that is not in mapping → ValueError.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        in_key: str = "demographics",
        out_dict_key: str = "demo_dict",
        out_values_key: str = "demo_values",
        out_missing_key: str = "demo_missing",
        allow_missing_keys: bool = True,
    ) -> None:
        # Register in_key with MapTransform, but we fully override __call__.
        super().__init__(keys=(in_key,), allow_missing_keys=allow_missing_keys)

        self.in_key = in_key
        self.out_dict_key = out_dict_key
        self.out_values_key = out_values_key
        self.out_missing_key = out_missing_key
        self.allow_missing_keys = allow_missing_keys

        # ---- Parse and validate config ----
        if "fields" not in config:
            raise ValueError("DemographicsConditioningd: config must contain a 'fields' mapping")

        raw_fields_cfg = config["fields"]
        if not isinstance(raw_fields_cfg, Mapping):
            raise TypeError("DemographicsConditioningd: config['fields'] must be a mapping")

        ordered_fields = config.get("ordered_fields")
        if ordered_fields is None:
            raise TypeError("DemographicsConditioningd: config['ordered_fields'] must be a provided")

        self.ordered_fields: list[str] = list(ordered_fields)
        if not self.ordered_fields:
            raise ValueError("DemographicsConditioningd: ordered_fields must be a non-empty list")

        # index mapping field_name -> slot in demo_values/demo_missing
        self.field_indices: dict[str, int] = {
            name: i for i, name in enumerate(self.ordered_fields)
        }

        # Storage for per-field config (normalized)
        self.fields_cfg: dict[str, dict[str, Any]] = {}

        # Lists by kind (for debugging / future use)
        self.numeric_fields: list[str] = []
        self.categorical_fields: list[str] = []
        self.ynd_fields: list[str] = []

        for name in self.ordered_fields:
            if name not in raw_fields_cfg:
                raise KeyError(
                    f"DemographicsConditioningd: field '{name}' is listed in "
                    "ordered_fields but not defined in config['fields']"
                )
            fcfg_raw = raw_fields_cfg[name]
            if not isinstance(fcfg_raw, Mapping):
                raise TypeError(f"Field config for '{name}' must be a mapping")

            fcfg = dict(fcfg_raw)
            kind = fcfg.get("kind")
            if kind not in ("numeric", "categorical", "ynd"):
                raise ValueError(
                    f"Field '{name}' has unsupported kind '{kind}' "
                    "(expected 'numeric', 'categorical', or 'ynd')"
                )
            fcfg["kind"] = kind

            if kind == "numeric":
                if "mean" not in fcfg or "std" not in fcfg:
                    raise ValueError(f"Numeric field '{name}' must define 'mean' and 'std'")
                mean = float(fcfg["mean"])
                std = float(fcfg["std"])
                if std <= 0.0:
                    raise ValueError(
                        f"Numeric field '{name}' must have std > 0, got {std}"
                    )
                fcfg["_mean"] = mean
                fcfg["_std"] = std

                na_values = fcfg.get("na_values")
                # default: treat "n/a" as missing if not overridden
                if na_values is None:
                    na_values = ["n/a"]
                fcfg["_na_values"] = [str(v) for v in na_values]

                self.numeric_fields.append(name)

            else:  # categorical or YND
                mapping = fcfg.get("mapping")
                if not isinstance(mapping, Mapping) or not mapping:
                    raise ValueError(
                        f"Categorical/YND field '{name}' must define non-empty 'mapping' dict"
                    )

                # Strict: only values explicitly listed are allowed.
                # Store mapping as str -> int.
                str2id: dict[str, int] = {}
                for raw_val, idx in mapping.items():
                    sval = str(raw_val).strip().lower()
                    ival = int(idx)
                    if sval in str2id:
                        raise ValueError(
                            f"Duplicate mapping key '{sval}' for field '{name}'"
                        )
                    str2id[sval] = ival

                fcfg["_mapping"] = str2id

                na_values = fcfg.get("na_values")
                if na_values is None:
                    # default: if "n/a" in mapping, use that as NA sentinel
                    if "n/a" in str2id:
                        na_values = ["n/a"]
                    else:
                        na_values = []
                fcfg["_na_values"] = [str(v) for v in na_values]
                fcfg["_na_token"] = fcfg["_na_values"][0] if fcfg["_na_values"] else None

                replace = fcfg.get("replace", None)
                if replace is None:
                    fcfg["_replace"] = {}
                else:
                    if not isinstance(replace, Mapping):
                        raise TypeError(f"Field '{name}' replace must be a mapping")
                    # Normalize keys/values to lowercase strings
                    rep: dict[str, str] = {}
                    for k, v in replace.items():
                        rep[str(k).strip().lower()] = str(v).strip().lower()
                    fcfg["_replace"] = rep



                if kind == "categorical":
                    self.categorical_fields.append(name)
                else:
                    self.ynd_fields.append(name)

            self.fields_cfg[name] = fcfg

        self.num_fields = len(self.ordered_fields)


        # Build numeric bins from age_group labels like "45-50"
        # and sort them by lower bound so we don't rely on YAML order.
        if "age_group" in self.fields_cfg:
            ag_cfg = self.fields_cfg["age_group"]
            if ag_cfg["kind"] == "categorical":
                mapping = ag_cfg.get("_mapping")
                if isinstance(mapping, Mapping) and mapping:
                    na_values = ag_cfg.get("_na_values", [])

                    range_bins: list[tuple[float, float, str]] = []
                    for lab in mapping.keys():
                        if lab in na_values or "-" not in lab:
                            continue
                        low_str, high_str = lab.split("-", 1)
                        try:
                            low = float(low_str.strip())
                            high = float(high_str.strip())
                        except Exception as exc:
                            raise ValueError(
                                f"Cannot convert age_group label '{lab}' into "
                                "low/high floats."
                            ) from exc
                        range_bins.append((low, high, lab))

                    range_bins.sort(key=lambda x: x[0])  # sort by low bound
                    self.age_group_bins = range_bins
    
    # per-field helpers
    def _handle_numeric(self, name: str, raw_demo: Mapping[str, Any]) -> (float, bool):
        cfg = self.fields_cfg[name]
        raw_val = raw_demo.get(name, None)

        # Completely missing field → treat as missing
        if raw_val is None:
            return 0.0, True

        sval = str(raw_val).strip()
        if sval == "" or sval in cfg["_na_values"]:
            # treat as missing; impute mean (0 after z-score)
            return 0.0, True

        try:
            v = float(sval)
        except Exception as e:
            raise ValueError(
                f"Numeric field '{name}' has non-numeric value {raw_val!r}"
            ) from e

        norm = (v - cfg["_mean"]) / cfg["_std"]
        return float(norm), False

    def _handle_categorical(self, name: str, raw_demo: Mapping[str, Any]) -> (int, bool):
        cfg = self.fields_cfg[name]
        raw_val = raw_demo.get(name, None)

        if raw_val is None:
            sval = ""
        else:
            sval = str(raw_val).strip().lower()
        
        rep = cfg.get("_replace", {})
        if sval in rep:
            sval = rep[sval]

        # missing field → NA token if defined
        if sval == "":
            if cfg["_na_token"] is None:
                raise ValueError(
                    f"Categorical field '{name}' missing in sample and no na_token defined"
                )
            idx = cfg["_mapping"][cfg["_na_token"]]
            return idx, True

        # Strict: only known keys allowed
        if sval not in cfg["_mapping"]:
            raise ValueError(
                f"Categorical field '{name}' got unexpected value {raw_val!r}; "
                f"allowed: {sorted(cfg['_mapping'].keys())}"
            )

        idx = cfg["_mapping"][sval]
        is_missing = sval in cfg["_na_values"]
        return int(idx), is_missing

    
    # derive age_group from age if needed
    def _derive_age_group(self, raw_demo: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        If config has both 'age' (numeric) and 'age_group' (categorical),
        and the sample has a valid age but no age_group, derive age_group
        by binning age into the configured range labels, e.g. '45-50'.
        """
        # Config must define both fields
        if "age" not in self.fields_cfg or "age_group" not in self.fields_cfg:
            return raw_demo

        age_cfg = self.fields_cfg["age"]
        age_group_cfg = self.fields_cfg["age_group"]

        # Types must match expectations
        if age_cfg["kind"] != "numeric" or age_group_cfg["kind"] != "categorical":
            return raw_demo

        # If we never built bins (no range-like labels), nothing to do
        if not getattr(self, "age_group_bins", None):
            return raw_demo

        # If dataset already provides age_group, respect it
        existing = raw_demo.get("age_group", None)
        if existing is not None and str(existing).strip() != "":
            return raw_demo

        # Need a usable age to derive from
        raw_age = raw_demo.get("age", None)
        if raw_age is None:
            return raw_demo

        sval = str(raw_age).strip()
        if sval == "" or sval in age_cfg["_na_values"]:
            # can't derive a bin from missing age
            return raw_demo

        try:
            age = float(sval)
        except Exception as exc:
            raise ValueError(
                f"Cannot convert age '{sval}' into float for deriving age_group."
            ) from exc

        na_values = age_group_cfg.get("_na_values", [])
        na_token = na_values[0] if na_values else None

        chosen_label = None
        for i, (low, high, lab) in enumerate(self.age_group_bins):
            is_last = (i == len(self.age_group_bins) - 1)
            # [low, high) for all but last; last bin is [low, high]
            if (age >= low and age < high) or (is_last and age >= low and age <= high):
                chosen_label = lab
                break

        # If outside all ranges, treat as NA if possible
        if chosen_label is None:
            if na_token is None:
                return raw_demo
            chosen_label = na_token

        new_demo = dict(raw_demo)
        new_demo["age_group"] = chosen_label
        return new_demo


    # main call
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)

        if self.in_key not in d:
            if self.allow_missing_keys:
                return d
            raise KeyError(
                f"DemographicsConditioningd: missing '{self.in_key}' in data"
            )

        raw_demo = d.get(self.in_key)
        if raw_demo is None:
            if self.allow_missing_keys:
                return d
            raise ValueError(
                f"DemographicsConditioningd: key '{self.in_key}' present but value is None"
            )

        if not isinstance(raw_demo, Mapping):
            raise TypeError(
                f"DemographicsConditioningd: expected '{self.in_key}' to be a mapping, "
                f"got {type(raw_demo)}"
            )

        # Derive age_group from age
        raw_demo = self._derive_age_group(raw_demo)

        demo_dict: dict[str, Any] = {}
        values: list[float] = []
        missing: list[bool] = []

        for name in self.ordered_fields:
            cfg = self.fields_cfg[name]
            if cfg["kind"] == "numeric":
                val, is_missing = self._handle_numeric(name, raw_demo)
            else:
                val, is_missing = self._handle_categorical(name, raw_demo)

            demo_dict[name] = val
            values.append(float(val))
            missing.append(bool(is_missing))

        d[self.out_dict_key] = demo_dict
        d[self.out_values_key] = torch.tensor(values, dtype=torch.float32)
        d[self.out_missing_key] = torch.tensor(missing, dtype=torch.bool)
        
        # Drop the raw nested demographics dict
        d.pop(self.in_key, None)
        
        return d
