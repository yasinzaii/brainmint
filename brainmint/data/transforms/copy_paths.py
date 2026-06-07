from collections.abc import Hashable, Sequence
from typing import Any

from monai.transforms import MapTransform


class CopyKeysToNewKeysd(MapTransform):
    """
    Copy values from src_keys to dst_keys. Intended to run BEFORE LoadImaged,
    so values are typically file-path strings.

    Example:
      src_keys=["latent"] -> dst_keys=["latent_path"]
      src_keys=["image","latent"] -> dst_keys=["image_path","latent_path"]
    """

    def __init__(
        self,
        src_keys: Sequence[Hashable],
        dst_keys: Sequence[Hashable] | None = None,
        postfix: str = "_path",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=list(src_keys), allow_missing_keys=allow_missing_keys)

        self.src_keys: list[Hashable] = list(src_keys)

        if dst_keys is not None:
            if len(dst_keys) != len(self.src_keys):
                raise ValueError(
                    f"dst_keys must have same length as src_keys. "
                    f"Got len(src_keys)={len(self.src_keys)} len(dst_keys)={len(dst_keys)}"
                )
            self.dst_keys: list[Hashable] = list(dst_keys)
        else:
            self.dst_keys = [f"{k}{postfix}" for k in self.src_keys]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)
        for src, dst in zip(self.src_keys, self.dst_keys, strict=True):
            if src not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Missing key '{src}' for CopyKeysToNewKeysd.")
            d[dst] = d[src]
        return d
