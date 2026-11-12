"""Live checkout lanes and the current alerts."""
from __future__ import annotations

import psycopg2
import psycopg2.extras
from fastapi import APIRouter

from ..db import _conn, _http_error
from ..settings import CHECKOUT_CAM_ID, LANE_MAX_CAPACITY

router = APIRouter()


def _lane_status(avg_wait_min: float, queue_depth: int) -> str:
    if avg_wait_min > 7 or queue_depth >= 8:
        return "busy_high"
    elif avg_wait_min > 4 or queue_depth >= 4:
        return "busy"
    elif queue_depth > 0 or avg_wait_min > 0:
        return "open"
    return "closed"


@router.get("/live-lanes")
def live_lanes():
    """
    Returns per-lane queue status for the Live tab.
    - Per-lane counts from queue_state_snapshots.lane_counts (head detector, live)
    - avg_wait_min from dashboard_state.service_min (stable, floor-capped)
    """
    try:
        import json
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT queue_count, lane_counts
                    FROM queue_state_snapshots
                    WHERE camera_id = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (CHECKOUT_CAM_ID,))
                snap_row = cur.fetchone()

                cur.execute("SELECT queue_now, service_min, wait_5m FROM dashboard_state WHERE id = 1")
                ds = cur.fetchone()

        total_queue = int(snap_row["queue_count"] or 0) if snap_row else (int(ds["queue_now"] or 0) if ds else 0)
        # Use 5-min forecast wait; fall back to service_min
        avg_wait_min = round(float(ds["wait_5m"]) if ds and ds["wait_5m"] is not None else float(ds["service_min"] if ds and ds["service_min"] else 1.0), 2)

        lane_counts_raw = snap_row["lane_counts"] if snap_row else None
        if isinstance(lane_counts_raw, str):
            lane_counts_raw = json.loads(lane_counts_raw)
        lane_counts: dict = lane_counts_raw or {}

        lanes = []
        for i in range(4):
            # Head-Detector writes lane_counts keyed 1..4 (matching lane1..lane4),
            # not 0-indexed — was reading str(i) here, which only ever matched
            # keys "1".."3" shifted by one lane and never read lane 4 at all.
            depth = int(lane_counts.get(str(i + 1), 0))
            avg_wait = avg_wait_min if depth > 0 else 0.0
            status = _lane_status(avg_wait, depth)
            lanes.append({
                "lane_number": i + 1,
                "lane_id":     str(i),
                "status":      status,
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

    except Exception as exc:
        raise _http_error(exc)


@router.get("/alerts")
def get_alerts():
    """Returns the current alert level using dashboard_state (same source as forecast tab)."""
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT wait_15m, updated_at
                    FROM dashboard_state WHERE id = 1
                """)
                state = cur.fetchone()

        if not state or state["wait_15m"] is None:
            return {"level": None, "message": "No forecast data yet.", "predicted_wait_min": None, "horizon_min": 15}

        wait = float(state["wait_15m"])

        if wait > 10:
            level, message = "red",    f"Queue exceeding {wait:.0f} min in 15 min — open a lane immediately."
        elif wait > 7:
            level, message = "orange", f"Queue building to {wait:.0f} min in 15 min — open a lane soon."
        elif wait > 5:
            level, message = "yellow", f"Queue may reach {wait:.0f} min — consider opening a lane."
        else:
            level, message = None,     f"Queue normal. Expected wait in 15 min: {wait:.1f} min."

        return {
            "level":              level,
            "message":            message,
            "predicted_wait_min": round(wait, 1),
            "horizon_min":        15,
        }

    except Exception as exc:
        raise _http_error(exc)
