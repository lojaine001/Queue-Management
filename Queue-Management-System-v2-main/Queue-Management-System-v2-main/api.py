"""
api.py — FastAPI backend for the IQMS Manager Mobile App.

Serves live queue status, alert level, and records manager responses.
Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="IQMS Manager API", version="1.0")

# Allow requests from the Expo app (any origin during development)
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


def _conn():
    return psycopg2.connect(**DB_CONFIG)


# ── Models ────────────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    response: str  # "opening_lane" | "cannot_open" | "false_alarm"
    lane_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/queue-status")
def queue_status():
    """
    Returns live queue depth and average dwell per lane.
    Reads the most recent queue_state_snapshots row per camera.
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT DISTINCT ON (camera_id)
                        camera_id,
                        queue_count,
                        avg_dwell_sec,
                        max_dwell_sec,
                        timestamp
                    FROM queue_state_snapshots
                    WHERE camera_id NOT LIKE 'SIM_%%'
                      AND timestamp >= NOW() - INTERVAL '24 hours'
                    ORDER BY camera_id, timestamp DESC
                """)
                rows = cur.fetchall()

                lanes = []
                for row in rows:
                    avg_wait_min = round((row["avg_dwell_sec"] or 0) / 60, 1)
                    queue_count = row["queue_count"] or 0
                    if avg_wait_min > 10 or queue_count >= 6:
                        status = "red"
                    elif avg_wait_min > 7 or queue_count >= 4:
                        status = "orange"
                    elif avg_wait_min > 5 or queue_count >= 2:
                        status = "yellow"
                    else:
                        status = "ok"

                    lanes.append({
                        "id": row["camera_id"],
                        "queue_depth": queue_count,
                        "avg_wait_min": avg_wait_min,
                        "max_wait_min": round((row["max_dwell_sec"] or 0) / 60, 1),
                        "status": status,
                        "last_updated": row["timestamp"].isoformat() if row["timestamp"] else None,
                    })

        return {"lanes": lanes, "fetched_at": datetime.now(timezone.utc).isoformat()}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/alerts")
def get_alerts():
    """
    Returns the current alert level based on the latest queue_predictions.
    level: null | "yellow" | "orange" | "red"
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get the most recent prediction
                cur.execute("""
                    SELECT est_wait_minutes, prediction_for, predicted_at
                    FROM queue_predictions
                    ORDER BY predicted_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()

        if not row:
            return {"level": None, "message": "No prediction data yet.", "predicted_wait_min": None}

        wait = float(row["est_wait_minutes"] or 0)
        horizon_min = None
        if row["prediction_for"] and row["predicted_at"]:
            delta = row["prediction_for"] - row["predicted_at"]
            horizon_min = max(0, int(delta.total_seconds() / 60))

        if wait > 10:
            level = "red"
            message = f"CRITICAL: Predicted {wait:.1f} min wait. All available to checkout."
        elif wait > 7:
            level = "orange"
            message = f"URGENT: Predicted {wait:.1f} min wait{f' in {horizon_min} min' if horizon_min else ''}. Open lane NOW."
        elif wait > 5:
            level = "yellow"
            message = f"Queue building: predicted {wait:.1f} min wait{f' in {horizon_min} min' if horizon_min else ''}. Open lane soon."
        else:
            level = None
            message = f"Queue normal. Predicted wait: {wait:.1f} min."

        return {
            "level": level,
            "message": message,
            "predicted_wait_min": round(wait, 1),
            "horizon_min": horizon_min,
            "predicted_at": row["predicted_at"].isoformat() if row["predicted_at"] else None,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/day-recap")
def day_recap():
    """
    Returns a simple summary of today's activity:
    total customers, peak hour, and average wait time.
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Total customers today
                cur.execute("""
                    SELECT COUNT(*) AS total
                    FROM entrance_events
                    WHERE timestamp >= CURRENT_DATE
                      AND camera_id NOT LIKE 'SIM_%%'
                """)
                total = int((cur.fetchone() or {}).get("total") or 0)

                # Peak hour today
                cur.execute("""
                    SELECT DATE_TRUNC('hour', timestamp) AS hour, COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE timestamp >= CURRENT_DATE
                      AND camera_id NOT LIKE 'SIM_%%'
                    GROUP BY 1
                    ORDER BY 2 DESC
                    LIMIT 1
                """)
                peak_row = cur.fetchone()
                peak_hour  = peak_row["hour"].strftime("%H:%M") if peak_row and peak_row["hour"] else None
                peak_count = int(peak_row["cnt"]) if peak_row else 0

                # Average wait today (from service_events)
                cur.execute("""
                    SELECT ROUND(AVG(total_dwell_sec) / 60.0, 1) AS avg_min
                    FROM service_events
                    WHERE timestamp >= CURRENT_DATE
                      AND camera_id NOT LIKE 'SIM_%%'
                """)
                avg_row = cur.fetchone()
                avg_wait_min = float(avg_row["avg_min"]) if avg_row and avg_row["avg_min"] else 0.0

        return {
            "total_customers": total,
            "peak_hour":       peak_hour,
            "peak_count":      peak_count,
            "avg_wait_min":    avg_wait_min,
            "date":            datetime.now().strftime("%d %b %Y"),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/alert-response")
def post_alert_response(body: AlertResponse):
    """
    Records the manager's response to an alert.
    response: "opening_lane" | "cannot_open" | "false_alarm"
    """
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