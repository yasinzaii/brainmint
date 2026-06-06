from typing import Any, Dict, Mapping, List, Optional, Tuple

import torch
import torch.nn as nn


class DemographicsEncoder(nn.Module):
    """
    Encode demographics into a fixed-size vector for diffusion conditioning.

    Expects as input (batch):
        demo_values:  (B, F) float32
        demo_missing: (B, F) bool

    where:
      - F = len(config['ordered_fields'])
      - config is the same one used by DemographicsConditioningd.

    Semantics (per field i):
      - kind == "numeric":
          demo_values[:, idx] = z-scored scalar
          (we pass this scalar directly)
      - kind == "categorical":
          demo_values[:, idx] = integer code (float-cast)
          (we cast to long and feed into nn.Embedding)
      - kind == "ynd":
          same as categorical, typically 3-way {"n/a":0, "no":1, "yes":2}

    We build:
        dem_raw = concat(
            numeric scalars,             # in ordered_fields order (numeric subset)
            all categorical embeddings,  # in ordered_fields order (categorical subset)
            all YND embeddings,          # in ordered_fields order (ynd subset)
            missing flags (as floats, exactly ordered_fields order)
        )  # shape: (B, raw_dim)
        dem_emb = MLP(dem_raw)  # -> (B, dem_embed_dim)
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        dem_embed_dim: int,
        default_cat_embed_dim: int = 8,
        default_ynd_embed_dim: int = 3,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if "fields" not in config:
            raise ValueError("DemographicsEncoder: config must contain a 'fields' mapping")

        fields_cfg = config["fields"]
        if not isinstance(fields_cfg, Mapping):
            raise TypeError("DemographicsEncoder: config['fields'] must be a mapping")

        ordered_fields = config.get("ordered_fields")
        if ordered_fields is None:
            raise TypeError(
                "DemographicsEncoder: config['ordered_fields'] must be provided and non-empty"
            )
        self.ordered_fields: List[str] = list(ordered_fields)
        if not self.ordered_fields:
            raise ValueError("DemographicsEncoder: ordered_fields must be non-empty")

        # map field_name -> index in demo_values/demo_missing
        self.field_indices: Dict[str, int] = {
            name: i for i, name in enumerate(self.ordered_fields)
        }

        # Keep track of which indices are numeric vs categorical vs YND
        self.numeric_indices: List[int] = []
        self.cat_fields: List[Tuple[str, int]] = []  # (name, index in demo_values)
        self.ynd_fields: List[Tuple[str, int]] = []  # (name, index in demo_values)

        cat_embeddings: Dict[str, nn.Embedding] = {}
        ynd_embeddings: Dict[str, nn.Embedding] = {}

        for name in self.ordered_fields:
            if name not in fields_cfg:
                raise KeyError(
                    f"DemographicsEncoder: field '{name}' is in ordered_fields "
                    "but not defined in config['fields']"
                )

            fcfg = fields_cfg[name]
            if not isinstance(fcfg, Mapping):
                raise TypeError(
                    f"DemographicsEncoder: field config for '{name}' must be a mapping"
                )

            kind = fcfg.get("kind")
            if kind not in ("numeric", "categorical", "ynd"):
                raise ValueError(
                    f"DemographicsEncoder: field '{name}' has unsupported kind '{kind}' "
                    "(expected 'numeric', 'categorical', or 'ynd')"
                )

            idx = self.field_indices[name]

            if kind == "numeric":
                self.numeric_indices.append(idx)
                continue

            # categorical / ynd
            mapping = fcfg.get("mapping")
            if not isinstance(mapping, Mapping) or not mapping:
                raise ValueError(
                    f"DemographicsEncoder: field '{name}' of kind '{kind}' "
                    "must define a non-empty 'mapping' dict"
                )

            # Collect and sort unique IDs
            raw_ids = sorted({int(v) for v in mapping.values()})
            
            # Enforce 0..K-1 contiguity
            expected = list(range(len(raw_ids)))
            if raw_ids != expected:
                raise ValueError(
                    f"DemographicsEncoder: field '{name}' mapping values must be "
                    f"contiguous 0..K-1, got {raw_ids}"
                )
            
            num_classes = len(raw_ids)

            # Allow per-field override of embedding dim; else default:
            if "embed_dim" in fcfg:
                emb_dim = int(fcfg["embed_dim"])
            else:
                if kind == "categorical":
                    emb_dim = min(default_cat_embed_dim, num_classes)
                else:  # "ynd"
                    emb_dim = default_ynd_embed_dim

            if emb_dim <= 0:
                raise ValueError(
                    f"DemographicsEncoder: field '{name}' has non-positive embed_dim={emb_dim}"
                )

            if kind == "categorical":
                self.cat_fields.append((name, idx))
                cat_embeddings[name] = nn.Embedding(
                    num_embeddings=num_classes,
                    embedding_dim=emb_dim,
                )
            else:  # kind == "ynd"
                self.ynd_fields.append((name, idx))
                ynd_embeddings[name] = nn.Embedding(
                    num_embeddings=num_classes,
                    embedding_dim=emb_dim,
                )

        self.cat_embeddings = nn.ModuleDict(cat_embeddings)
        self.ynd_embeddings = nn.ModuleDict(ynd_embeddings)

        # Compute raw feature dimension
        num_numeric = len(self.numeric_indices)
        cat_dim = sum(emb.embedding_dim for emb in self.cat_embeddings.values())
        ynd_dim = sum(emb.embedding_dim for emb in self.ynd_embeddings.values())
        flags_dim = len(self.ordered_fields)  # one missing flag per field

        raw_dim = num_numeric + cat_dim + ynd_dim + flags_dim
        if raw_dim == 0:
            raise ValueError("DemographicsEncoder: raw_dim=0; no features configured?")

        if hidden_dim is None:
            hidden_dim = dem_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dem_embed_dim),
        )

    def forward(
        self,
        demo_values: torch.Tensor,
        demo_missing: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            demo_values:  (B, F) float32, produced by DemographicsConditioningd
            demo_missing: (B, F) bool,   produced by DemographicsConditioningd

        Returns:
            dem_emb: (B, dem_embed_dim) float32
        """
        if demo_values.ndim != 2:
            raise ValueError(
                f"DemographicsEncoder: demo_values must be 2D (B,F), "
                f"got {tuple(demo_values.shape)}"
            )
        if demo_missing.ndim != 2:
            raise ValueError(
                f"DemographicsEncoder: demo_missing must be 2D (B,F), "
                f"got {tuple(demo_missing.shape)}"
            )
        if demo_values.shape != demo_missing.shape:
            raise ValueError(
                "DemographicsEncoder: demo_values and demo_missing must have the same shape; "
                f"got {tuple(demo_values.shape)} vs {tuple(demo_missing.shape)}"
            )

        B, F = demo_values.shape
        if F != len(self.ordered_fields):
            raise ValueError(
                f"DemographicsEncoder: demo_values has F={F}, but config has "
                f"{len(self.ordered_fields)} ordered_fields"
            )

        parts: List[torch.Tensor] = []

        # 1) numeric scalars (z-scored, already prepared by the transform)
        if self.numeric_indices:
            x_num = demo_values[:, self.numeric_indices]  # (B, N_num)
            parts.append(x_num)

        # 2) categorical embeddings
        if self.cat_fields:
            cat_embs: List[torch.Tensor] = []
            for name, idx in self.cat_fields:
                ids = demo_values[:, idx].long()  # indices encoded by transform
                cat_embs.append(self.cat_embeddings[name](ids))
            x_cat = torch.cat(cat_embs, dim=-1)  # (B, sum cat dims)
            parts.append(x_cat)

        # 3) YND embeddings
        if self.ynd_fields:
            ynd_embs: List[torch.Tensor] = []
            for name, idx in self.ynd_fields:
                ids = demo_values[:, idx].long()
                ynd_embs.append(self.ynd_embeddings[name](ids))
            x_ynd = torch.cat(ynd_embs, dim=-1)  # (B, sum ynd dims)
            parts.append(x_ynd)

        # 4) Missing flags (one per field, exactly ordered_fields order)
        x_flags = demo_missing.float()  # (B, F)
        parts.append(x_flags)

        dem_raw = torch.cat(parts, dim=-1)  # (B, raw_dim)
        dem_emb = self.mlp(dem_raw)         # (B, dem_embed_dim)
        return dem_emb
