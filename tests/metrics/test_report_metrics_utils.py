"""Unit tests for helper utilities in the report metrics script."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics.autoencoder import report_metrics as rm


def _split_cells(line: str) -> List[str]:
    return [cell.strip() for cell in line.split("|")]


def test_to_list_handles_various_inputs() -> None:
    assert rm._to_list(None) == []
    assert rm._to_list([1, 2]) == [1, 2]
    assert rm._to_list(("a", "b")) == ["a", "b"]
    assert rm._to_list(torch.tensor(5)) == [5]
    tensor_list = rm._to_list(torch.tensor([[1, 2], [3, 4]]))
    assert tensor_list == [[1, 2], [3, 4]]
    assert rm._to_list("value") == ["value"]


def test_string_list_decodes_bytes_and_none() -> None:
    result = rm._string_list([b"abc", None, 42])
    assert result == ["abc", "", "42"]


def test_format_metric_handles_numeric_and_special_values() -> None:
    assert rm._format_metric("ssim", 0.98765) == "0.9877"
    assert rm._format_metric("psnr", 31.2345) == "31.23"
    assert rm._format_metric("unknown", 1.234567) == "1.2346"
    assert rm._format_metric("mse", None) == "nan"
    assert rm._format_metric("mse", float("inf")) == "inf"
    assert rm._format_metric("mse", float("nan")) == "nan"
    assert rm._format_metric("mse", "text") == "text"


def test_report_metadata_row_prefix_formats_modality() -> None:
    metadata = rm.ReportMetadata(model_name="Autoencoder", study_name="Study", compression="1/4")
    assert metadata.row_prefix("t1w") == ["Autoencoder (t1w)", "Study", "1/4"]
    assert metadata.row_prefix("") == ["Autoencoder", "Study", "1/4"]


def test_build_table_formats_rows() -> None:
    metadata = rm.ReportMetadata(model_name="Autoencoder", study_name="Brain Study", compression="1/4")
    rows = [
        {"modality": "t1w", "lpips": 0.12345, "ssim": 0.9876, "psnr": 31.5, "mse": 0.00123},
        {"modality": "t2w", "lpips": 0.22345, "ssim": 0.8876, "psnr": 29.25, "mse": 0.00234},
    ]

    table = rm._build_table(rows, metadata)
    assert len(table) == 4
    header_cells = _split_cells(table[0])
    assert header_cells == [
        "Model (Architecture)",
        "Study",
        "Compression (Linear)",
        "LPIPS",
        "SSIM",
        "PSNR",
        "MSE",
    ]

    first_row = _split_cells(table[2])
    assert first_row == [
        "Autoencoder (t1w)",
        "Brain Study",
        "1/4",
        "0.1235",
        "0.9876",
        "31.50",
        "0.001230",
    ]

    second_row = _split_cells(table[3])
    assert second_row == [
        "Autoencoder (t2w)",
        "Brain Study",
        "1/4",
        "0.2235",
        "0.8876",
        "29.25",
        "0.002340",
    ]


def test_summary_rows_handle_missing_metrics() -> None:
    aggregator = rm.MetricAggregator(metric_names=("lpips", "ssim"))
    aggregator.add_sample("t1w", {"lpips": 0.5})
    aggregator.add_sample("t1w", {"lpips": 0.7, "ssim": 0.8})

    rows = aggregator.summary_rows()
    assert rows[0]["modality"] == "T1W"
    assert math.isclose(rows[0]["lpips"], 0.6)
    assert math.isclose(rows[0]["ssim"], 0.8)
    assert rows[1]["modality"] == "Overall"
