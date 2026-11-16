"""The Live tab: each checkout lane's queue and status, and the current alert level."""
from __future__ import annotations

import json

import psycopg2
import psycopg2.extras

from ..db import _conn
from ..repositories import dashboard, snapshots
from ..settings import CHECKOUT_CAM_ID, LANE_MAX_CAPACITY

LANES = 4


def lane_status(avg_wait_min: float, queue_depth: int) -> str:
    if avg_wait_min > 7 or queue_depth >= 8:
        return "busy_high"
    elif avg_wait_min > 4 or queue_depth >= 4:
        return "busy"
    elif queue_depth > 0 or avg_wait_min > 0:
        return "open"
    return "closed"


def live_lanes() -> dict:
    """
    Per-lane queue status for the Live tab.
    - Per-lane counts from the checkout camera's newest snapshot (lane_counts, live)
    - avg_wait_min: the forecast wait in 5 minutes; service_min before the first
      forecast; 1.0 when neither exists
    """
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            snap_row = snapshots.latest(cur, CHECKOUT_CAM_ID)
            ds = dashboard.live_inputs(cur)

    total_queue = int(snap_row["queue_count"] or 0) if snap_row else (int(ds["queue_now"] or 0) if ds else 0)
    # Use 5-min forecast wait; fall back to service_min
    avg_wait_min = round(float(ds["wait_5m"]) if ds and ds["wait_5m"] is not None else float(ds["service_min"] if ds and ds["service_min"] else 1.0), 2)

    lane_counts_raw = snap_row["lane_counts"] if snap_row else None
    if isinstance(lane_counts_raw, str):
        lane_counts_raw = json.loads(lane_counts_raw)
    lane_counts: dict = lane_counts_raw or {}

    lanes = []
    for i in range(LANES):
        # Head-Detector writes lane_counts keyed 1..4 (matching lane1..lane4),
        # not 0-indexed — was reading str(i) here, which only ever matched
        # keys "1".."3" shifted by one lane and never read lane 4 at all.
        depth = int(lane_counts.get(str(i + 1), 0))
        avg_wait = avg_wait_min if depth > 0 else 0.0
        lanes.append({
            "lane_number": i + 1,
            "lane_id":     str(i),
            "status":      lane_status(avg_wait, depth),
            "waiting":     depth,
            "fill":        depth,
            "fill_max":    LANE_MAX_CAPACITY,
            "avg_wait_min": avg_wait,
        })

    snapshot = {
        "total_in_queue": total_queue,
        "avg_wait_min":   avg_wait_min,
        "open_lanes":     len([l for l in lanes if l["status"] != "closed"]),
    }
    return {"lanes": lanes, "snapshot": snapshot}


def alert_level(wait):
    """(level, message) for a forecast wait in 15 minutes: red > 10, orange > 7, yellow > 5."""
    if wait > 10:
        return "red",    f"Queue exceeding {wait:.0f} min in 15 min — open a lane immediately."
    elif wait > 7:
        return "orange", f"Queue building to {wait:.0f} min in 15 min — open a lane soon."
    elif wait > 5:
        return "yellow", f"Queue may reach {wait:.0f} min — consider opening a lane."
    return None,         f"Queue normal. Expected wait in 15 min: {wait:.1f} min."


def current_alert() -> dict:
    """The alert level from the published forecast (same source as the forecast tab)."""
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            state = dashboard.wait_in_15_min(cur)

    if not state or state["wait_15m"] is None:
        return {"level": None, "message": "No forecast data yet.", "predicted_wait_min": None, "horizon_min": 15}

    wait = float(state["wait_15m"])
    level, message = alert_level(wait)
    return {
        "level":              level,
        "message":            message,
        "predicted_wait_min": round(wait, 1),
        "horizon_min":        15,
    }
