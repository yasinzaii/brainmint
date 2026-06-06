from typing import Any, Dict, Mapping, Optional
import torch
from monai.transforms import MapTransform, Transform

class MapModalityToLabeld(MapTransform):
    """Read 'modality' (str) and write <key_name> (torch.long) using a 0-based mapping."""
    def __init__(
        self,
        mapping: Mapping[str, int],
        allow_missing_keys: bool = False,
        fail_on_unknown: bool = True,
        key_name: Optional[str] = None
    ):
        super().__init__(keys=("modality",), allow_missing_keys=allow_missing_keys)
        self.mapping = dict(mapping)
        self.fail_on_unknown = bool(fail_on_unknown)
        self.key_name = key_name

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if "modality" not in d:
            return d  # nothing to do
        m = str(d["modality"])
        if m not in self.mapping:
            if self.fail_on_unknown:
                raise KeyError(f"Unknown modality '{m}'. Known: {list(self.mapping.keys())}")
            idx = 0 
        else:
            idx = int(self.mapping[m])
        d[self.key_name] = torch.tensor(idx, dtype=torch.long)
        return d


class ConstantModalityLabeld(Transform):
    """Write a constant modality label tensor to ``key_name``."""
    def __init__(
        self,
        label: int,
        key_name: str = "modality_map",
        overwrite: bool = True,
    ):
        self.label = int(label)
        self.key_name = key_name
        self.overwrite = bool(overwrite)
        print(f"Conditioning 'ConstantModalityLabeld', Const Target is: {label}")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if not self.overwrite and self.key_name in d:
            return d
        d[self.key_name] = torch.tensor(self.label, dtype=torch.long)
        return d
