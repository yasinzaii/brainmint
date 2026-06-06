from __future__ import annotations

import pytest

from brainmint.utils.schedules import (
    PiecewiseSchedule,
    build_piecewise_lambda,
    build_piecewise_linear_lambda,
)


def test_piecewise_schedule_steps_and_linear_segments_merge() -> None:
    sched = PiecewiseSchedule(
        [
            {"type": "step", "epoch": 0, "value": {"prob": {"real": 1.0, "synthetic": 0.0}}},
            {
                "type": "linear",
                "start_epoch": 2,
                "end_epoch": 6,
                "start_value": {"prob": {"synthetic": 0.0}},
                "end_value": {"prob": {"synthetic": 0.8}},
            },
        ],
        name="choice",
    )

    assert sched.value_at(0) == {"prob": {"real": 1.0, "synthetic": 0.0}}
    epoch4 = sched.value_at(4)
    assert epoch4["prob"]["real"] == 1.0
    assert epoch4["prob"]["synthetic"] == pytest.approx(0.4)
    assert sched.value_at(9) == {"prob": {"real": 1.0, "synthetic": 0.8}}


def test_piecewise_lambda_is_stepwise_constant() -> None:
    fn = build_piecewise_lambda([(0, 1.0), (5, 0.5), (10, 0.1)])

    assert fn(0) == 1.0
    assert fn(4) == 1.0
    assert fn(5) == 0.5
    assert fn(99) == 0.1


def test_piecewise_linear_lambda_interpolates_between_points() -> None:
    fn = build_piecewise_linear_lambda([(0, 1.0), (10, 0.0)])

    assert fn(0) == 1.0
    assert fn(5) == pytest.approx(0.5)
    assert fn(10) == 0.0
    assert fn(11) == 0.0


def test_empty_scheduler_specs_raise() -> None:
    with pytest.raises(ValueError):
        build_piecewise_lambda([])
    with pytest.raises(ValueError):
        build_piecewise_linear_lambda([])
