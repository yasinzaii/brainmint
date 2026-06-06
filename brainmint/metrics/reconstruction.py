"""Metric utilities for evaluating autoencoder reconstructions."""


import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from monai.losses import PerceptualLoss
from monai.metrics import MSEMetric, PSNRMetric, SSIMMetric, MultiScaleSSIMMetric
from monai.utils import MetricReduction



class ReconstructionMetricCalculator:
    """Compute LPIPS, SSIM, MS-SSIM, PSNR and MSE for reconstructed MRI volumes.

    Args:
        max_value: Upper bound of the expected data range.
        lpips_network: Backbone passed to MONAI's :class:`PerceptualLoss`.
        lpips_is_fake_3d: Whether to run LPIPS in fake-3D mode when inputs are 3D.
        lpips_fake_3d_ratio: Ratio argument forwarded to MONAI when fake-3D is on.
        clamp_inputs: Clamp predictions and targets into ``[0, max_value]`` before
            dispatching range-sensitive metrics (PSNR, SSIM and LPIPS). The MSE
            calculation always observes the raw tensors so large absolute errors
            are still visible even when clamping is enabled. When ``False`` we
            validate the tensors and raise an error if they contain values
            outside the configured range so callers can surface upstream
            normalisation bugs instead of silently saturating the data.
        ssim_kernel_type: Optional override for MONAI's SSIM kernel shape.
        ssim_kernel_sigma: Optional override for MONAI's SSIM Gaussian sigma.
        ssim_k1: Optional override for the luminance stabiliser used by SSIM.
        ssim_k2: Optional override for the contrast stabiliser used by SSIM.
            MS-SSIM shares these kernel and stability settings so overrides apply
            to both metrics for consistency with MONAI's API.
    """

    METRIC_NAMES: Tuple[str, ...] = ("lpips", "ssim", "ms_ssim", "psnr", "mse")

    def __init__(
        self,
        max_value: float = 1.0,
        lpips_network: str = "alex",
        lpips_is_fake_3d: bool = True,
        lpips_fake_3d_ratio: float = 0.5,
        clamp_inputs: bool = True,
        ssim_kernel_type: str | None = None,
        ssim_kernel_sigma: float | Sequence[float] | None = None,
        ssim_k1: float | None = None,
        ssim_k2: float | None = None,
    ) -> None:
        self.max_value = float(max_value)
        self.lpips_network = str(lpips_network)
        self.lpips_is_fake_3d = bool(lpips_is_fake_3d)
        self.lpips_fake_3d_ratio = float(lpips_fake_3d_ratio)
        self.clamp_inputs = bool(clamp_inputs)
        self.ssim_kernel_type = ssim_kernel_type
        self.ssim_kernel_sigma = ssim_kernel_sigma
        self.ssim_k1 = ssim_k1
        self.ssim_k2 = ssim_k2

        reduction = MetricReduction.NONE
        self.mse_metric = MSEMetric(reduction=reduction)
        self.psnr_metric = PSNRMetric(max_val=self.max_value, reduction=reduction)
        self._ssim_metric: SSIMMetric | None = None
        self._ssim_dims: int | None = None
        self._ssim_win_size: int | None = None
        self._ms_ssim_metric: MultiScaleSSIMMetric | None = None
        self._ms_ssim_dims: int | None = None
        self._ms_ssim_win_size: int | None = None
        self._lpips_metric: PerceptualLoss | None = None
        self._lpips_dims: int | None = None
        self._lpips_device: torch.device | None = None

    @property
    def metric_names(self) -> Tuple[str, ...]:
        return self.METRIC_NAMES

    def compute_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> List[Dict[str, float]]:
        """Return metric dictionaries for each item in the batch."""

        if predictions.shape != targets.shape:
            raise ValueError(
                "Predictions and targets must have identical shapes; "
                f"got {predictions.shape} and {targets.shape}."
            )
        if predictions.ndim < 3:
            raise ValueError("Expected tensors with batch, channel and spatial dimensions.")

        preds = predictions.float()
        refs = targets.float()

        self._validate_numerics(preds, refs)

        if self.clamp_inputs:
            preds_in_range = preds.clamp(0.0, self.max_value)
            refs_in_range = refs.clamp(0.0, self.max_value)
        else:
            self._validate_data_range(preds, refs)
            preds_in_range = preds
            refs_in_range = refs


        batch_size = preds.shape[0]
        spatial_dims = preds.ndim - 2

        if spatial_dims != 3:
            raise ValueError(f"Expected 3D Images (3 Dimensions)! But number of dimensions = {spatial_dims}.")

        self._ensure_ssim_metric(spatial_dims, preds.shape[2:])
        self._ensure_ms_ssim_metric(spatial_dims, preds.shape[2:])
        self._ensure_lpips_metric(spatial_dims, preds.device)

        mse_vals = self._compute_regression_metric(self.mse_metric, preds, refs)
        psnr_vals = self._compute_regression_metric(self.psnr_metric, preds_in_range, refs_in_range)
        ssim_vals = self._compute_ssim(preds_in_range, refs_in_range)
        ms_ssim_vals = self._compute_ms_ssim(preds_in_range, refs_in_range)
        lpips_vals = self._compute_lpips(preds_in_range, refs_in_range)

        batch_metrics: List[Dict[str, float]] = []
        for idx in range(batch_size):
            batch_metrics.append(
                {
                    "lpips": float(lpips_vals[idx]),
                    "ssim": float(ssim_vals[idx]),
                    "ms_ssim": float(ms_ssim_vals[idx]),
                    "psnr": float(psnr_vals[idx]),
                    "mse": float(mse_vals[idx]),
                }
            )
        return batch_metrics

    def _ensure_ssim_metric(self, spatial_dims: int, spatial_shape: Sequence[int]) -> None:
        win_size = self._determine_ssim_window(spatial_shape)
        if (
            self._ssim_metric is None
            or self._ssim_dims != spatial_dims
            or self._ssim_win_size != win_size
        ):
            ssim_kwargs = dict(
                spatial_dims=spatial_dims,
                data_range=self.max_value,
                win_size=win_size,
                reduction=MetricReduction.NONE,
            )
            if self.ssim_kernel_type is not None:
                ssim_kwargs["kernel_type"] = self.ssim_kernel_type
            if self.ssim_kernel_sigma is not None:
                ssim_kwargs["kernel_sigma"] = self.ssim_kernel_sigma
            if self.ssim_k1 is not None:
                ssim_kwargs["k1"] = self.ssim_k1
            if self.ssim_k2 is not None:
                ssim_kwargs["k2"] = self.ssim_k2

            self._ssim_metric = SSIMMetric(**ssim_kwargs)
            self._ssim_dims = spatial_dims
            self._ssim_win_size = win_size

    def _ensure_ms_ssim_metric(self, spatial_dims: int, spatial_shape: Sequence[int]) -> None:
        win_size = self._determine_ms_ssim_window(spatial_shape)
        if (
            self._ms_ssim_metric is None
            or self._ms_ssim_dims != spatial_dims
            or self._ms_ssim_win_size != win_size
        ):
            ms_kwargs = dict(
                spatial_dims=spatial_dims,
                data_range=self.max_value,
                kernel_size=win_size,
                reduction=MetricReduction.NONE,
            )
            if self.ssim_kernel_type is not None:
                ms_kwargs["kernel_type"] = self.ssim_kernel_type
            if self.ssim_kernel_sigma is not None:
                ms_kwargs["kernel_sigma"] = self.ssim_kernel_sigma
            if self.ssim_k1 is not None:
                ms_kwargs["k1"] = self.ssim_k1
            if self.ssim_k2 is not None:
                ms_kwargs["k2"] = self.ssim_k2

            self._ms_ssim_metric = MultiScaleSSIMMetric(**ms_kwargs)
            self._ms_ssim_dims = spatial_dims
            self._ms_ssim_win_size = win_size

    @staticmethod
    def _determine_ssim_window(spatial_shape: Sequence[int]) -> int:
        if not spatial_shape:
            raise ValueError("Spatial dimensions required to configure SSIM metric.")
        min_dim = min(int(dim) for dim in spatial_shape)
        if min_dim < 3:
            raise ValueError(
                f"SSIM requires spatial dims ≥ 3 in every axis; got min spatial size {min_dim}."
            )
        max_kernel = min(11, min_dim)
        if max_kernel % 2 == 0:
            max_kernel = max(1, max_kernel - 1)
        if max_kernel < 3 and min_dim >= 3:
            max_kernel = 3
        return max_kernel

    @staticmethod
    def _determine_ms_ssim_window(spatial_shape: Sequence[int]) -> int:
        """Choose an MS-SSIM kernel size that is compatible with the image size.

        MONAI's `compute_ms_ssim` downsamples the image multiple times (one per
        entry in the `weights` sequence). With the default 5-scale weights and
        an 11x11x11 kernel this requires fairly large spatial dimensions; 
        Otherwise MONAI raises a ValueError
        """
        if not spatial_shape:
            raise ValueError("Spatial dimensions required to configure MS-SSIM metric.")
        
        min_dim = min(int(dim) for dim in spatial_shape)
        
       
        num_scales = 5  # NOTE: DEFAULT MS-SSIM Settings
        weights_div = max(1, (num_scales - 1)) ** 2 
        max_kernel_from_ms = max(1, min_dim // weights_div)

        # 11 is the default max. 
        max_kernel = min(11, max_kernel_from_ms)

        # force odd kernel size
        if max_kernel % 2 == 0:
            max_kernel = max(1, max_kernel - 1)

        if max_kernel < 3:
            raise ValueError(
                f"MS-SSIM cannot be computed for spatial size {spatial_shape}: "
                f"minimum dimension {min_dim} is too small for the default "
                f"5-scale MS-SSIM. Consider disabling MS-SSIM or using fewer "
                f"scales / a smaller kernel."
            )
        return max_kernel

    def _ensure_lpips_metric(self, spatial_dims: int, device: torch.device) -> None:
        if self._lpips_metric is not None and self._lpips_dims == spatial_dims:
            if self._lpips_device != device:
                self._lpips_metric.to(device)
                self._lpips_device = device
            return

        if spatial_dims == 3:
            is_fake_3d = self.lpips_is_fake_3d
            fake_ratio = self.lpips_fake_3d_ratio
        else:
            is_fake_3d = False
            fake_ratio = 1.0

        self._lpips_metric = PerceptualLoss(
            spatial_dims=spatial_dims,
            network_type=self.lpips_network,
            is_fake_3d=is_fake_3d,
            fake_3d_ratio=fake_ratio,
        )
        self._lpips_metric.eval()
        self._lpips_metric.to(device)
        self._lpips_dims = spatial_dims
        self._lpips_device = device

    def _compute_regression_metric(
        self, metric: MSEMetric | PSNRMetric, preds: torch.Tensor, refs: torch.Tensor
    ) -> torch.Tensor:
        metric(y_pred=preds, y=refs)
        values = metric.aggregate()
        metric.reset()
        batch_size = preds.shape[0]
        return self._finalise_metric_values(values, batch_size)

    def _compute_ssim(self, preds: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        if self._ssim_metric is None:
            raise RuntimeError("SSIM metric has not been initialised.")
        self._ssim_metric(y_pred=preds, y=refs)
        values = self._ssim_metric.aggregate()
        self._ssim_metric.reset()
        batch_size = preds.shape[0]
        return self._finalise_metric_values(values, batch_size)

    def _compute_ms_ssim(self, preds: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        if self._ms_ssim_metric is None:
            raise RuntimeError("MS-SSIM metric has not been initialised.")
        self._ms_ssim_metric(y_pred=preds, y=refs)
        values = self._ms_ssim_metric.aggregate()
        self._ms_ssim_metric.reset()
        batch_size = preds.shape[0]
        return self._finalise_metric_values(values, batch_size)

    def _compute_lpips(self, preds: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        if self._lpips_metric is None:
            raise RuntimeError("LPIPS metric has not been initialised.")

        lpips_preds, lpips_refs = self._prepare_lpips_inputs(preds, refs)
        device = self._lpips_device or preds.device
        lpips_preds = lpips_preds.to(device)
        lpips_refs = lpips_refs.to(device)

        results: List[float] = []
        with torch.no_grad():
            for idx in range(lpips_preds.shape[0]):
                value = self._lpips_metric(lpips_preds[idx : idx + 1], lpips_refs[idx : idx + 1])
                results.append(float(value.detach().cpu()))

        return self._finalise_metric_values(torch.tensor(results, dtype=torch.float32), batch_size=lpips_preds.shape[0])

    def _prepare_lpips_inputs(
        self, preds: torch.Tensor, refs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalise tensors into the [-1, 1] range expected by LPIPS."""
        preds_lpips = preds.detach().clone()
        refs_lpips = refs.detach().clone()
        repeat_dims = [1] * preds_lpips.ndim
        if preds_lpips.shape[1] == 1:
            # MONAI's PerceptualLoss implementations all consume 3-channel inputs. If the
            # reconstructions are single-channel volumes we broadcast them to RGB so the
            # downstream network sees identical information in each channel.
            repeat_dims[1] = 3
            preds_lpips = preds_lpips.repeat(*repeat_dims)
            refs_lpips = refs_lpips.repeat(*repeat_dims)
        elif preds_lpips.shape[1] != 3:
            raise ValueError(
                "LPIPS metric expects inputs with either 1 or 3 channels; "
                f"received {preds_lpips.shape[1]}."
            )

        # MONAI's PerceptualLoss is trained on ImageNet-style images that live in [-1, 1].
        # Our tensors are already guaranteed to be inside [0, max_value] at this point, so
        # divide by the configured range and shift into the expected interval without any
        # extra clamping that could hide upstream scaling bugs.
        mv = float(self.max_value) if self.max_value > 0 else 1.0
        scale = 2.0 / mv
        preds_lpips = preds_lpips * scale - 1.0
        refs_lpips = refs_lpips * scale - 1.0
        return preds_lpips, refs_lpips

    def _finalise_metric_values(
        self, values: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        values = values.detach().cpu()
        if values.shape[0] != batch_size:
            raise ValueError(
                "Metric output batch dimension does not match inputs; "
                f"expected {batch_size}, got {values.shape[0]}."
            )
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.squeeze(dim=1)
        elif values.ndim > 1 and any(dim != 1 for dim in values.shape[1:]):
            raise ValueError(
                "Expected MONAI metric to return per-sample scalars; "
                f"received shape {tuple(values.shape)}."
            )
        return values

    def _validate_data_range(self, preds: torch.Tensor, refs: torch.Tensor) -> None:
        """Raise if tensors fall outside ``[0, max_value]``."""

        min_val = min(float(preds.min().item()), float(refs.min().item()))
        max_val = max(float(preds.max().item()), float(refs.max().item()))
        lower_bound = -1e-6
        upper_bound = float(self.max_value) + 1e-6
        if min_val < lower_bound or max_val > upper_bound:
            raise ValueError(
                "Metric inputs must already be scaled to the [0, max_value] data range. "
                "Either normalise upstream or set clamp_inputs=True to clip them."
            )

    def _validate_numerics(self, preds: torch.Tensor, refs: torch.Tensor) -> None:
        tensors = (preds, refs)
        names = ("predictions", "targets")
        for tensor, name in zip(tensors, names):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name.capitalize()} contain NaN or Inf values.")


class MetricAggregator:
    """Aggregate metrics across modalities and compute summary statistics."""

    def __init__(
        self, metric_names: Sequence[str], modalities: Sequence[str] | None = None
    ) -> None:
        self.metric_names = tuple(metric_names)
        self._storage: MutableMapping[str, MutableMapping[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._counts: MutableMapping[str, int] = defaultdict(int)
        self._order: List[str] = []
        self._overall_key = "overall"
        self._storage[self._overall_key]

        if modalities is not None:
            for modality in modalities:
                key = self._normalise_modality(modality)
                if key != self._overall_key and key not in self._order:
                    self._order.append(key)

        self._seen_modalities = set(self._order)

    @staticmethod
    def _normalise_modality(modality: str) -> str:
        return str(modality).strip().lower() or "unknown"

    @staticmethod
    def _format_modality(modality: str) -> str:
        key = modality.strip().lower()
        if key == "overall":
            return "Overall"
        if not key:
            return "Unknown"
        if len(key) <= 3:
            return key.upper()
        return key.capitalize()

    def add_sample(self, modality: str, metrics: Mapping[str, float]) -> None:
        key = self._normalise_modality(modality)
        if key != self._overall_key and key not in self._seen_modalities:
            self._order.append(key)
            self._seen_modalities.add(key)

        bucket = self._storage[key]
        for name in self.metric_names:
            if name in metrics:
                value = float(metrics[name])
                bucket[name].append(value)
                if key != self._overall_key:
                    self._storage[self._overall_key][name].append(value)
            else:
                # Skip metrics that were not provided for this sample so downstream
                # aggregation can still proceed with partially populated records.
                continue
        self._counts[key] += 1
        if key != self._overall_key:
            self._counts[self._overall_key] += 1

    def summary_rows(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        ordered_keys = list(self._order) + [self._overall_key]
        seen: set[str] = set()
        for key in ordered_keys:
            if key in seen:
                continue
            seen.add(key)
            metrics = self._storage.get(key, {})
            row: Dict[str, float] = {
                "modality": self._format_modality(key),
                "count": int(self._counts.get(key, 0)),
            }
            for name in self.metric_names:
                row[name] = self._summarise(metrics.get(name, []))
            rows.append(row)
        return rows

    @staticmethod
    def _summarise(values: Iterable[float]) -> float:
        vals = list(values)
        if not vals:
            return float("nan")
        finite = [v for v in vals if math.isfinite(v)]
        if finite:
            return float(np.mean(finite))
        if any(math.isinf(v) for v in vals):
            return float("inf")
        return float("nan")
