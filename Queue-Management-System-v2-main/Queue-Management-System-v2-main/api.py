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
STORE_TZ = os.getenv("STORE_TZ", "Europe/Paris")


def _conn():
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute("SET timezone = %s", (STORE_TZ,))
    conn.commit()
    return conn


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
                # Recent activity per lane — use last 2 hours as fallback
                cur.execute("""
                    SELECT
                        lane_id,
                        COUNT(*)                                    AS recent_checkouts,
                        ROUND(AVG(total_dwell_sec)::numeric / 60.0, 1)      AS avg_wait_min
                    FROM service_events
                    WHERE timestamp >= NOW() - INTERVAL '2 hours'
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND lane_id IS NOT NULL
                    GROUP BY lane_id
                    ORDER BY lane_id
                """)
                lane_rows = cur.fetchall()

                # Read queue count and avg wait from dashboard_state — these are
                # the exact values the dashboard computes and displays, so the
                # app snapshot matches the dashboard perfectly.
                cur.execute("""
                    SELECT queue_now AS queue_count,
                           service_min * 60 AS avg_dwell_sec,
                           open_lanes
                    FROM dashboard_state WHERE id = 1
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
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/forecast")
def forecast():
    """
    Reads the pre-computed values that the dashboard already calculates and
    writes to dashboard_state on every refresh. This guarantees the app shows
    exactly the same numbers as the dashboard — no separate computation needed.
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT queue_now, service_min,
                           wait_0m, wait_5m, wait_10m, wait_15m,
                           lane1_wait_15m, lane2_wait_15m, lane3_wait_15m, lane4_wait_15m,
                           open_lanes, updated_at
                    FROM dashboard_state
                    WHERE id = 1
                """)
                state = cur.fetchone()

        def _f(v): return round(float(v), 1) if v is not None else None

        if not state:
            # dashboard_state not populated yet — dashboard hasn't run since last restart
            return {
                "wait_now_min": None, "wait_5_min": None,
                "wait_10_min": None, "wait_15_min": None,
                "current_lanes": 1, "lane_scenarios": [],
            }

        current_lanes = int(state["open_lanes"] or 1)
        wait_now = _f(state["wait_0m"])
        wait_5   = _f(state["wait_5m"])
        wait_10  = _f(state["wait_10m"])
        wait_15  = _f(state["wait_15m"])

        lane_waits = {
            1: _f(state["lane1_wait_15m"]) or 0.0,
            2: _f(state["lane2_wait_15m"]) or 0.0,
            3: _f(state["lane3_wait_15m"]) or 0.0,
            4: _f(state["lane4_wait_15m"]) or 0.0,
        }

        scenarios = []
        for n in range(1, 5):
            estimated = lane_waits[n]
            if estimated > 10:
                color = "red"
            elif estimated > 7:
                color = "orange"
            elif estimated > 4:
                color = "yellow"
            else:
                color = "green"
            scenarios.append({
                "lanes":        n,
                "est_wait_min": estimated,
                "color":        color,
                "is_current":   n == current_lanes,
            })

        return {
            "wait_now_min":   round(wait_now, 1),
            "wait_5_min":     wait_5,
            "wait_10_min":    wait_10,
            "wait_15_min":    wait_15,
            "current_lanes":  current_lanes,
            "lane_scenarios": scenarios,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/day-recap")
def day_recap():
    """Returns today's summary (falls back to most recent day with data)."""
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Find the most recent day that has data
                cur.execute("""
                    SELECT DATE(timestamp) AS day
                    FROM entrance_events
                    WHERE camera_id NOT LIKE 'SIM_%%'
                    ORDER BY timestamp DESC LIMIT 1
                """)
                day_row = cur.fetchone()
                ref_date = day_row["day"] if day_row else None
                date_filter = f"DATE(timestamp) = '{ref_date}'" if ref_date else "timestamp >= CURRENT_DATE"
                display_date = ref_date.strftime("%d %b %Y") if ref_date else datetime.now().strftime("%d %b %Y")

                cur.execute(f"""
                    SELECT COUNT(*) AS total
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                """)
                total = int((cur.fetchone() or {}).get("total") or 0)

                cur.execute(f"""
                    SELECT DATE_TRUNC('hour', timestamp) AS hour, COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                    GROUP BY 1 ORDER BY 2 DESC LIMIT 1
                """)
                peak_row   = cur.fetchone()
                peak_hour  = peak_row["hour"].strftime("%H:%M") if peak_row and peak_row["hour"] else None
                peak_end   = (peak_row["hour"] + timedelta(hours=1)).strftime("%H:%M") if peak_row and peak_row["hour"] else None
                peak_count = int(peak_row["cnt"]) if peak_row else 0

                cur.execute(f"""
                    SELECT ROUND(AVG(total_dwell_sec)::numeric / 60.0, 1) AS avg_min
                    FROM service_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                """)
                avg_row      = cur.fetchone()
                avg_wait_min = float(avg_row["avg_min"]) if avg_row and avg_row["avg_min"] else 0.0

                # Equipment mix
                cur.execute(f"""
                    SELECT equipment_type, COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND equipment_type IS NOT NULL
                      AND equipment_type != 'none'
                    GROUP BY equipment_type
                """)
                equip_rows = cur.fetchall()

                # Demographics and hourly entries read from dashboard_state
                # so values match the dashboard exactly.
                cur.execute("""
                    SELECT demographics_json, entries_hour_json
                    FROM dashboard_state WHERE id = 1
                """)
                ds_row = cur.fetchone()

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

        # Demographics and hourly from dashboard_state (exact match with dashboard)
        import json as _json
        _demo      = _json.loads(ds_row["demographics_json"])   if ds_row and ds_row["demographics_json"]   else {}
        _hourly    = _json.loads(ds_row["entries_hour_json"])   if ds_row and ds_row["entries_hour_json"]   else []

        return {
            "date":                display_date,
            "total_customers":     total,
            "avg_wait_min":        avg_wait_min,
            "peak_hour":           peak_hour,
            "peak_hour_end":       peak_end,
            "peak_count":          peak_count,
            "equipment":           equipment,
            "demographics_gender": _demo.get("gender", []),
            "demographics_age":    _demo.get("age",    []),
            "entries_by_hour":     _hourly,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/forecast-chart")
def forecast_chart():
    """Returns 60-min time series of predicted arrivals and wait for the app chart."""
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT DISTINCT ON (prediction_for)
                        prediction_for,
                        COALESCE(ensemble_yhat, 0)    AS arrivals,
                        COALESCE(est_wait_minutes, 0) AS wait_min
                    FROM queue_predictions
                    WHERE prediction_for >= NOW()
                      AND prediction_for <= NOW() + INTERVAL '60 minutes'
                    ORDER BY prediction_for ASC, predicted_at DESC
                """)
                rows = cur.fetchall()

        return {
            "slots": [
                {
                    "time":     row["prediction_for"].strftime("%H:%M"),
                    "arrivals": round(float(row["arrivals"]), 1),
                    "wait_min": round(float(row["wait_min"]), 1),
                }
                for row in rows
            ]
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