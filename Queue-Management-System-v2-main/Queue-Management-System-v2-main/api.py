"""
api.py — FastAPI backend for the IQMS Manager Mobile App.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="IQMS Manager API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "iqms"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "0000"),
)

LANE_MAX_CAPACITY = 10  # denominator for fill bar


def _conn():
    return psycopg2.connect(**DB_CONFIG)


def _lane_status(avg_wait_min: float, queue_depth: int) -> str:
    if avg_wait_min > 7 or queue_depth >= 8:
        return "busy_high"
    elif avg_wait_min > 4 or queue_depth >= 4:
        return "busy"
    elif queue_depth > 0 or avg_wait_min > 0:
        return "open"
    return "closed"


# ── Models ────────────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    response: str
    lane_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/live-lanes")
def live_lanes():
    """
    Returns per-lane queue status for the Live tab.
    Reads recent service_events grouped by lane_id + current queue snapshot.
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Recent activity per lane (last 20 min)
                cur.execute("""
                    SELECT
                        lane_id,
                        COUNT(*)                                    AS recent_checkouts,
                        ROUND(AVG(total_dwell_sec) / 60.0, 1)      AS avg_wait_min
                    FROM service_events
                    WHERE timestamp >= NOW() - INTERVAL '20 minutes'
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND lane_id IS NOT NULL
                    GROUP BY lane_id
                    ORDER BY lane_id
                """)
                lane_rows = cur.fetchall()

                # Current total queue snapshot
                cur.execute("""
                    SELECT DISTINCT ON (camera_id)
                        queue_count, avg_dwell_sec
                    FROM queue_state_snapshots
                    WHERE camera_id NOT LIKE 'SIM_%%'
                      AND timestamp >= NOW() - INTERVAL '5 minutes'
                    ORDER BY camera_id, timestamp DESC
                    LIMIT 1
                """)
                snap = cur.fetchone()
                total_queue = int(snap["queue_count"] or 0) if snap else 0
                active_lanes = max(len(lane_rows), 1)
                queue_per_lane = max(0, round(total_queue / active_lanes))

        lanes = []
        active_lane_ids = {str(r["lane_id"]) for r in lane_rows}

        # Build up to 4 lanes
        known_lanes = sorted(active_lane_ids) if active_lane_ids else []
        # Pad to 4 if fewer
        while len(known_lanes) < 4:
            known_lanes.append(str(len(known_lanes)))

        for i, lane_id in enumerate(known_lanes[:4]):
            matching = next((r for r in lane_rows if str(r["lane_id"]) == lane_id), None)
            if matching:
                avg_wait = float(matching["avg_wait_min"] or 0)
                depth = queue_per_lane
                status = _lane_status(avg_wait, depth)
            else:
                avg_wait = 0.0
                depth = 0
                status = "closed"

            lanes.append({
                "lane_number": i + 1,
                "lane_id":     lane_id,
                "status":      status,
                "waiting":     depth,
                "fill":        depth,
                "fill_max":    LANE_MAX_CAPACITY,
                "avg_wait_min": avg_wait,
            })

        snapshot = {
            "total_in_queue": total_queue,
            "avg_wait_min":   round(float(snap["avg_dwell_sec"] or 0) / 60, 1) if snap else 0.0,
            "open_lanes":     len([l for l in lanes if l["status"] != "closed"]),
        }

        return {"lanes": lanes, "snapshot": snapshot}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/alerts")
def get_alerts():
    """Returns the current alert level based on the latest queue_predictions."""
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT est_wait_minutes, prediction_for, predicted_at
                    FROM queue_predictions
                    ORDER BY predicted_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()

        if not row:
            return {"level": None, "message": "No prediction data yet.", "predicted_wait_min": None, "horizon_min": None}

        wait = float(row["est_wait_minutes"] or 0)
        horizon_min = None
        if row["prediction_for"] and row["predicted_at"]:
            delta = row["prediction_for"] - row["predicted_at"]
            horizon_min = max(0, int(delta.total_seconds() / 60))

        if wait > 10:
            level, message = "red",    f"Queue exceeding {wait:.0f} min — open a lane immediately."
        elif wait > 7:
            level, message = "orange", f"Predicted {wait:.0f} min wait in {horizon_min or '?'} min. Open a lane soon."
        elif wait > 5:
            level, message = "yellow", f"Queue building: {wait:.0f} min predicted. Consider opening a lane."
        else:
            level, message = None, f"Queue normal. Predicted wait: {wait:.1f} min."

        return {
            "level":              level,
            "message":            message,
            "predicted_wait_min": round(wait, 1),
            "horizon_min":        horizon_min,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/forecast")
def forecast():
    """
    Returns predicted wait at +15 and +30 min, plus lane scenarios.
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                now = datetime.now(timezone.utc)

                # Get predictions for the next 60 min
                cur.execute("""
                    SELECT DISTINCT ON (DATE_TRUNC('minute', prediction_for))
                        prediction_for,
                        est_wait_minutes,
                        predicted_at
                    FROM queue_predictions
                    WHERE prediction_for BETWEEN NOW() AND NOW() + INTERVAL '65 minutes'
                    ORDER BY DATE_TRUNC('minute', prediction_for), predicted_at DESC
                """)
                predictions = cur.fetchall()

                # Current active lanes
                cur.execute("""
                    SELECT COUNT(DISTINCT lane_id) AS active_lanes
                    FROM service_events
                    WHERE timestamp >= NOW() - INTERVAL '20 minutes'
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND lane_id IS NOT NULL
                """)
                lane_row = cur.fetchone()
                current_lanes = max(int(lane_row["active_lanes"] or 1), 1) if lane_row else 1

        def closest_wait(target_min: int) -> Optional[float]:
            target_time = now + timedelta(minutes=target_min)
            if not predictions:
                return None
            closest = min(predictions, key=lambda r: abs((r["prediction_for"] - target_time).total_seconds()))
            return round(float(closest["est_wait_minutes"] or 0), 1)

        wait_now    = closest_wait(0)  or 0.0
        wait_15     = closest_wait(15)
        wait_30     = closest_wait(30)

        # Lane scenarios: wait scales inversely with lane count
        # Approximation: wait_N = wait_now * (current_lanes / N)
        scenarios = []
        for n in range(1, 6):
            estimated = round(wait_now * (current_lanes / n), 1) if wait_now > 0 else 0.0
            if estimated > 10:
                color = "red"
            elif estimated > 7:
                color = "orange"
            elif estimated > 4:
                color = "yellow"
            else:
                color = "green"
            scenarios.append({
                "lanes":         n,
                "est_wait_min":  estimated,
                "color":         color,
                "is_current":    n == current_lanes,
            })

        return {
            "wait_now_min":    round(wait_now, 1),
            "wait_15_min":     wait_15,
            "wait_30_min":     wait_30,
            "current_lanes":   current_lanes,
            "lane_scenarios":  scenarios,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/day-recap")
def day_recap():
    """Returns today's summary including equipment mix."""
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT COUNT(*) AS total
                    FROM entrance_events
                    WHERE timestamp >= CURRENT_DATE
                      AND camera_id NOT LIKE 'SIM_%%'
                """)
                total = int((cur.fetchone() or {}).get("total") or 0)

                cur.execute("""
                    SELECT DATE_TRUNC('hour', timestamp) AS hour, COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE timestamp >= CURRENT_DATE
                      AND camera_id NOT LIKE 'SIM_%%'
                    GROUP BY 1 ORDER BY 2 DESC LIMIT 1
                """)
                peak_row   = cur.fetchone()
                peak_hour  = peak_row["hour"].strftime("%H:%M") if peak_row and peak_row["hour"] else None
                peak_end   = (peak_row["hour"] + timedelta(hours=1)).strftime("%H:%M") if peak_row and peak_row["hour"] else None
                peak_count = int(peak_row["cnt"]) if peak_row else 0

                cur.execute("""
                    SELECT ROUND(AVG(total_dwell_sec) / 60.0, 1) AS avg_min
                    FROM service_events
                    WHERE timestamp >= CURRENT_DATE
                      AND camera_id NOT LIKE 'SIM_%%'
                """)
                avg_row      = cur.fetchone()
                avg_wait_min = float(avg_row["avg_min"]) if avg_row and avg_row["avg_min"] else 0.0

                # Equipment mix
                cur.execute("""
                    SELECT equipment_type, COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE timestamp >= CURRENT_DATE
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND equipment_type IS NOT NULL
                      AND equipment_type != 'none'
                    GROUP BY equipment_type
                """)
                equip_rows = cur.fetchall()

        equip_total = sum(int(r["cnt"]) for r in equip_rows) or 1
        equipment = []
        order = ["trolley", "store_basket", "personal_bag"]
        colors = {"trolley": "#06b6d4", "store_basket": "#a855f7", "personal_bag": "#f59e0b"}
        labels = {"trolley": "Trolley", "store_basket": "Store basket", "personal_bag": "Personal bag"}
        for key in order:
            row = next((r for r in equip_rows if r["equipment_type"] == key), None)
            count = int(row["cnt"]) if row else 0
            equipment.append({
                "type":    key,
                "label":   labels[key],
                "count":   count,
                "percent": round(count / equip_total * 100),
                "color":   colors[key],
            })

        return {
            "date":            datetime.now().strftime("%d %b %Y"),
            "total_customers": total,
            "avg_wait_min":    avg_wait_min,
            "peak_hour":       peak_hour,
            "peak_hour_end":   peak_end,
            "peak_count":      peak_count,
            "equipment":       equipment,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/alert-response")
def post_alert_response(body: AlertResponse):
    valid = {"opening_lane", "cannot_open", "false_alarm"}
    if body.response not in valid:
        raise HTTPException(status_code=400, detail=f"response must be one of {valid}")
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS alert_responses (
                        id SERIAL PRIMARY KEY,
                        response TEXT NOT NULL,
                        lane_id TEXT,
                        responded_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute(
                    "INSERT INTO alert_responses (response, lane_id) VALUES (%s, %s)",
                    (body.response, body.lane_id),
                )
            conn.commit()
        return {"status": "recorded", "response": body.response}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))