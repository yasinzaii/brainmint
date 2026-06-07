from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import torch

from brainmint.inference.core.context import InferenceContext
from brainmint.inference.core.interfaces import DiffusionPipeline, Postprocessor
from brainmint.utils.spatial import center_crop_or_pad_zyx


class MedDDPMGenerationPipeline(DiffusionPipeline):
    """Generation pipeline for Med-DDPM BraTS mask-conditioned synthesis."""

    required_modules: ClassVar[set[str]] = {"med_ddpm"}

    def __init__(
        self,
        *,
        condition_key: str = "med_ddpm_mask_one_hot",
        modality_key: str = "modality",
        output_size_zyx: Sequence[int] = (144, 192, 192),
        modality_to_channel: Mapping[str, int] | None = None,
        normalize_to_01: bool = True,
        postprocess: Postprocessor | None = None,
    ) -> None:
        super().__init__()
        self.condition_key = str(condition_key)
        self.modality_key = str(modality_key)
        self.output_size_zyx = tuple(int(value) for value in output_size_zyx)
        self.normalize_to_01 = bool(normalize_to_01)
        self.postprocess = postprocess

        default_map = {"t1w": 0, "t1ce": 1, "t2w": 2, "flair": 3}
        self.modality_to_channel = dict(default_map)
        if modality_to_channel:
            self.modality_to_channel.update({str(key).lower(): int(value) for key, value in modality_to_channel.items()})

    def run(self, batch: Mapping[str, Any], *, ctx: InferenceContext) -> dict[str, Any]:
        model = ctx.get("med_ddpm", required=True)
        if self.condition_key not in batch:
            raise KeyError(f"Batch missing condition_key='{self.condition_key}'.")

        condition = batch[self.condition_key]
        if not torch.is_tensor(condition):
            condition = torch.as_tensor(condition)
        condition = condition.to(device=ctx.device, dtype=ctx.dtype)
        if condition.ndim == 4:
            condition = condition.unsqueeze(0)
        if condition.ndim != 5:
            raise ValueError(f"Condition tensor must be (B,4,D,W,H), got {tuple(condition.shape)}")

        batch_size = int(condition.shape[0])
        all_modalities = model.sample_all(batch_size=batch_size, condition_tensors=condition)
        if not torch.is_tensor(all_modalities):
            all_modalities = torch.as_tensor(all_modalities, device=ctx.device, dtype=ctx.dtype)
        else:
            all_modalities = all_modalities.to(device=ctx.device, dtype=ctx.dtype)

        modalities = batch.get(self.modality_key)
        if modalities is None:
            raise KeyError(f"Batch missing modality_key='{self.modality_key}'")
        if isinstance(modalities, (list, tuple)):
            modality_list = list(modalities)
        elif torch.is_tensor(modalities):
            modality_list = [str(value) for value in modalities.tolist()]
        else:
            modality_list = [str(modalities)] * batch_size

        channel_indices = []
        for modality in modality_list:
            key = str(modality).lower()
            if key not in self.modality_to_channel:
                raise KeyError(f"Unknown modality '{modality}'. Known: {sorted(self.modality_to_channel)}")
            channel_indices.append(self.modality_to_channel[key])

        index_tensor = torch.tensor(channel_indices, device=ctx.device, dtype=torch.long)
        selected = all_modalities[torch.arange(batch_size, device=ctx.device), index_tensor].unsqueeze(1)

        if self.normalize_to_01:
            selected = ((selected + 1.0) / 2.0).clamp(0.0, 1.0)

        selected = center_crop_or_pad_zyx(selected, self.output_size_zyx)
        selected = selected.permute(0, 1, 4, 3, 2).contiguous()

        if self.postprocess is not None:
            selected = self.postprocess(selected, ctx=ctx)

        return {"sample": selected}
