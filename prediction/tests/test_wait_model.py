"""
Regression tests for compute_wait_estimates() lane-scenario behaviour.

Run with:  python -m pytest prediction/tests/test_wait_model.py -v
"""
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from prediction.core import compute_wait_estimates, BUCKET_MINUTES


def _make_forecast(n_steps: int = 20, arrivals_per_bucket: float = 4.0) -> pd.DataFrame:
    """Fixed-input forecast with constant arrivals — deterministic, no DB needed."""
    now = pd.Timestamp("2026-06-27 10:00:00")
    ts = [now + pd.Timedelta(minutes=BUCKET_MINUTES * i) for i in range(n_steps)]
    return pd.DataFrame({"ds": ts, "yhat": arrivals_per_bucket})


# ── Scenario comparison tests (no cap) ────────────────────────────────────────

def test_scenario_waits_strictly_decrease_across_lanes():
    """Lanes 1-4 must produce strictly decreasing wait_15m (uncapped mode)."""
    forecast = _make_forecast(arrivals_per_bucket=4.0)
    current_queue = 15
    dwell = 3.0
    waits = []
    for n in range(1, 5):
        _, w15, _, _ = compute_wait_estimates(
            forecast.copy(),
            current_queue=max(0, current_queue - n),
            avg_dwell_min=dwell,
            active_lanes=n,
            max_queue_per_lane=None,  # scenario mode — no cap
            max_wait_min=None,
        )
        waits.append(w15)

    for i in range(len(waits) - 1):
        assert waits[i] is not None and waits[i + 1] is not None, \
            f"wait_15m is None for scenario {i + 1} or {i + 2}"
        assert waits[i] > waits[i + 1], (
            f"Expected wait({i+1} lanes)={waits[i]:.2f} > wait({i+2} lanes)={waits[i+1]:.2f} "
            f"but got equal or reversed values. Lane factor is cancelling out."
        )


def test_scenario_waits_low_demand_still_decrease():
    """Even with low arrivals (light queue), more lanes should still reduce wait."""
    forecast = _make_forecast(arrivals_per_bucket=1.5)
    current_queue = 8
    dwell = 2.5
    waits = []
    for n in range(1, 5):
        _, w15, _, _ = compute_wait_estimates(
            forecast.copy(),
            current_queue=max(0, current_queue - n),
            avg_dwell_min=dwell,
            active_lanes=n,
            max_queue_per_lane=None,
            max_wait_min=None,
        )
        waits.append(w15)

    # At minimum: 1-lane wait must be greater than 4-lane wait
    assert waits[0] is not None and waits[3] is not None
    assert waits[0] > waits[3], (
        f"1-lane wait {waits[0]:.2f} should exceed 4-lane wait {waits[3]:.2f}"
    )


# ── Saturation behaviour (cap applied) ────────────────────────────────────────

def test_saturation_cap_causes_flat_waits():
    """
    Document the known saturation mode: when max_queue_per_lane is set and the
    queue saturates, the lane factor cancels and waits flatten.

    This test exists to make the behaviour EXPLICIT and detectable — if the cap
    is ever removed from production mode too, this test will fail and alert us.
    """
    forecast = _make_forecast(n_steps=30, arrivals_per_bucket=10.0)
    current_queue = 100  # very large — guaranteed to saturate
    dwell = 3.0
    max_per_lane = 20

    waits = []
    for n in range(1, 5):
        _, w15, _, _ = compute_wait_estimates(
            forecast.copy(),
            current_queue=max(0, current_queue - n),
            avg_dwell_min=dwell,
            active_lanes=n,
            max_queue_per_lane=max_per_lane,
            max_wait_min=None,
        )
        waits.append(w15)

    # Under saturation with the cap, all lanes hit max_per_lane * dwell.
    # We assert they are very close (within 1 minute) to document the flat behaviour.
    assert waits[0] is not None and waits[1] is not None
    assert abs(waits[0] - waits[1]) < 1.0, (
        f"Expected 1-lane ({waits[0]:.2f}) and 2-lane ({waits[1]:.2f}) to be nearly "
        f"identical under saturation (cap={max_per_lane}). If they now differ significantly "
        f"the cap may have been removed from production mode too."
    )


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_zero_queue_returns_zero_wait():
    forecast = _make_forecast(arrivals_per_bucket=0.0)
    _, w15, _, _ = compute_wait_estimates(
        forecast, current_queue=0, avg_dwell_min=3.0, active_lanes=2,
        max_queue_per_lane=None, max_wait_min=None,
    )
    assert w15 == 0.0


def test_empty_forecast_returns_none():
    empty = pd.DataFrame(columns=["ds", "yhat"])
    result, w15, w30, svc = compute_wait_estimates(
        empty, current_queue=5, avg_dwell_min=3.0, active_lanes=2,
    )
    assert result == []
    assert w15 is None
    assert w30 is None
