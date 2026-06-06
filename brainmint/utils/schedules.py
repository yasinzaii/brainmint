"""Training schedule builders and reusable epoch schedule primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "LinearSegment",
    "PiecewiseSchedule",
    "StepPoint",
    "build_piecewise_lambda",
    "build_piecewise_linear_lambda",
]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _deep_merge(a: Any, b: Any) -> Any:
    """Deep merge dict-like structures; ``b`` overrides ``a``."""

    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = dict(a)
        for key, b_value in b.items():
            out[key] = _deep_merge(out[key], b_value) if key in out else b_value
        return out
    return b


def _deep_lerp(a: Any, b: Any, t: float) -> Any:
    """Linearly interpolate nested numeric/dict structures."""

    if _is_number(a) and _is_number(b):
        return float(a) + (float(b) - float(a)) * float(t)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys = set(a.keys()) | set(b.keys())
        out: dict[str, Any] = {}
        for key in keys:
            if key not in a:
                out[str(key)] = b[key]
            elif key not in b:
                out[str(key)] = a[key]
            else:
                out[str(key)] = _deep_lerp(a[key], b[key], t)
        return out
    return b if t >= 0.5 else a


@dataclass(frozen=True)
class StepPoint:
    epoch: int
    value: Any


@dataclass(frozen=True)
class LinearSegment:
    start_epoch: int
    end_epoch: int
    start_value: Any
    end_value: Any


class PiecewiseSchedule:
    """Evaluate step and linear schedules over epochs.

    Supported entries:
      - ``type: step``: value becomes active at ``epoch >= entry.epoch``.
      - ``type: linear``: value is interpolated between start/end epochs and
        clamped after the segment ends.

    Step values provide the base value for an epoch. Linear segments active at
    that epoch are then merged in start-epoch order, which makes resumed runs
    evaluate to the full effective config rather than only the newest segment.
    """

    def __init__(self, spec: Optional[Sequence[Mapping[str, Any]]], *, name: str) -> None:
        self.name = name
        self.steps: list[StepPoint] = []
        self.lines: list[LinearSegment] = []
        if not spec:
            return

        for item in spec:
            if not isinstance(item, Mapping):
                raise TypeError(f"{name}: each schedule entry must be a mapping, got {type(item)}")
            typ = str(item.get("type", "step")).lower().strip()

            if typ == "step":
                if "epoch" not in item:
                    raise KeyError(f"{name}: step entry missing 'epoch'")
                if "value" not in item:
                    raise KeyError(f"{name}: step entry missing 'value'")
                self.steps.append(StepPoint(epoch=int(item["epoch"]), value=item["value"]))

            elif typ == "linear":
                if "start_epoch" not in item or "end_epoch" not in item:
                    raise KeyError(f"{name}: linear entry must include start_epoch and end_epoch")
                start_epoch = int(item["start_epoch"])
                end_epoch = int(item["end_epoch"])
                start_value = item.get("start_value", item.get("start", None))
                end_value = item.get("end_value", item.get("end", None))
                if start_value is None or end_value is None:
                    raise KeyError(f"{name}: linear entry must include start_value/end_value (or start/end)")
                self.lines.append(
                    LinearSegment(
                        start_epoch=start_epoch,
                        end_epoch=end_epoch,
                        start_value=start_value,
                        end_value=end_value,
                    )
                )
            else:
                raise ValueError(f"{name}: unknown schedule type {typ!r}")

        self.steps.sort(key=lambda point: point.epoch)
        self.lines.sort(key=lambda segment: segment.start_epoch)

    def _eval_line(self, segment: LinearSegment, epoch: int) -> Any:
        ep = int(epoch)
        if segment.end_epoch <= segment.start_epoch:
            return segment.end_value
        if ep <= segment.start_epoch:
            return segment.start_value
        if ep >= segment.end_epoch:
            return segment.end_value
        t = (ep - segment.start_epoch) / float(segment.end_epoch - segment.start_epoch)
        return _deep_lerp(segment.start_value, segment.end_value, t)

    def value_at(self, epoch: int) -> Optional[Any]:
        if not (self.steps or self.lines):
            return None

        ep = int(epoch)
        base = None
        for step in self.steps:
            if step.epoch <= ep:
                base = step.value
            else:
                break

        value = base
        for segment in self.lines:
            if segment.start_epoch <= ep:
                value = _deep_merge(value, self._eval_line(segment, ep))
            else:
                break

        return value



def build_piecewise_lambda(schedule: Sequence[Sequence[int | float]]):
    """Return ``lambda(epoch)`` for piecewise-constant scheduler factors."""

    if not schedule:
        raise ValueError("Empty schedule")

    points = [(int(epoch), float(value)) for epoch, value in schedule]

    def lr_fn(epoch: int) -> float:
        value = points[0][1]
        for start_epoch, factor in points:
            if epoch < start_epoch:
                break
            value = factor
        return value

    return lr_fn


def build_piecewise_linear_lambda(schedule: Sequence[Sequence[int | float]]):
    """Return ``lambda(epoch)`` that linearly interpolates scheduler factors."""

    if not schedule:
        raise ValueError("Empty schedule")

    points = sorted((int(epoch), float(value)) for epoch, value in schedule)

    def lr_lambda(epoch: int) -> float:
        if epoch <= points[0][0]:
            return points[0][1]

        for (start_epoch, start_val), (end_epoch, end_val) in zip(points[:-1], points[1:]):
            if epoch <= end_epoch:
                span = max(1, end_epoch - start_epoch)
                t = (epoch - start_epoch) / span
                return start_val + t * (end_val - start_val)

        return points[-1][1]

    return lr_lambda

