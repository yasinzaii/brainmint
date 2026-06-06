from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from brainmint.utils.state_dict_loader import load_module_state_dict, load_weight_specs


class Owner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)


def test_load_module_state_dict_from_root_checkpoint(tmp_path: Path) -> None:
    source = nn.Linear(2, 2)
    target = nn.Linear(2, 2)
    path = tmp_path / "root.pt"
    torch.save(source.state_dict(), path)

    loaded = load_module_state_dict(target, path=str(path), state_key="<root>", freeze=True)

    assert loaded is True
    assert target.training is False
    assert all(not p.requires_grad for p in target.parameters())
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_load_weight_specs_resolves_nested_owner_targets(tmp_path: Path) -> None:
    source = Owner()
    target = Owner()
    path = tmp_path / "nested.pt"
    torch.save({"state_dict": {"encoder": source.encoder.state_dict()}}, path)

    loaded = load_weight_specs(
        target,
        [{"target": "encoder", "path": str(path), "state_key": "encoder", "freeze": False}],
    )

    assert loaded == ["encoder"]
    assert all(p.requires_grad for p in target.encoder.parameters())
    for key, value in source.encoder.state_dict().items():
        assert torch.equal(target.encoder.state_dict()[key], value)


def test_optional_missing_checkpoint_is_skipped(tmp_path: Path) -> None:
    target = nn.Linear(2, 2)

    loaded = load_module_state_dict(
        target,
        path=str(tmp_path / "missing.pt"),
        state_key="<root>",
        optional=True,
    )

    assert loaded is False
