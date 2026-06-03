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

                # Aggregate latest snapshot across all real cameras.
                # Previously used DISTINCT ON + LIMIT 1 which always picked the
                # alphabetically-first camera (Bosch_Camera_Entrance) instead of
                # the checkout/Head-Detector camera (Bosch_Camera_exit).
                cur.execute("""
                    SELECT
                        COALESCE(SUM(queue_count), 0)  AS queue_count,
                        MAX(avg_dwell_sec)             AS avg_dwell_sec
                    FROM (
                        SELECT DISTINCT ON (camera_id)
                            queue_count, avg_dwell_sec
                        FROM queue_state_snapshots
                        WHERE camera_id NOT LIKE 'SIM_%%'
                        ORDER BY camera_id, timestamp DESC
                    ) latest_per_camera
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
                # Use DISTINCT ON to get the best (most recently generated) prediction
                # for the next upcoming slot. Avoids the pipeline.py single-row problem:
                # pipeline.py writes one row per call with a very fresh predicted_at,
                # so ORDER BY predicted_at DESC LIMIT 1 always returned that one row
                # whose est_wait_minutes = wait_15m (a far-future peak value ≈ 25 min).
                cur.execute("""
                    SELECT DISTINCT ON (prediction_for)
                        est_wait_minutes, wait_15m, prediction_for, predicted_at
                    FROM queue_predictions
                    WHERE prediction_for >= NOW()
                    ORDER BY prediction_for ASC, predicted_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()

        if not row:
            return {"level": None, "message": "No prediction data yet.", "predicted_wait_min": None, "horizon_min": None}

        # Use wait_15m (15-min lookahead) for the alert threshold — same as dashboard.
        # est_wait_minutes is the predicted wait for this slot; wait_15m is the
        # 15-minute horizon from this slot, which is what the manager needs to act on.
        wait = float(row["wait_15m"] or row["est_wait_minutes"] or 0)

        if wait > 10:
            level, message = "red",    f"Queue exceeding {wait:.0f} min — open a lane immediately."
        elif wait > 7:
            level, message = "orange", f"Queue building to {wait:.0f} min in 15 min. Open a lane soon."
        elif wait > 5:
            level, message = "yellow", f"Queue may reach {wait:.0f} min. Consider opening a lane."
        else:
            level, message = None, f"Queue normal. Predicted wait: {wait:.1f} min."

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
    Returns predicted wait at NOW/+5/+10/+15 min, plus lane scenarios.
    Re-simulates the queue on every call (like the dashboard does) so values
    are always based on current state, not stale est_wait_minutes from the DB.
    """
    BUCKET_MIN = 3  # ensemble model bucket size in minutes

    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

                # 1. Live queue snapshot
                cur.execute("""
                    SELECT
                        COALESCE(SUM(queue_count), 0) AS queue_count,
                        MAX(avg_dwell_sec)            AS avg_dwell_sec
                    FROM (
                        SELECT DISTINCT ON (camera_id)
                            queue_count, avg_dwell_sec
                        FROM queue_state_snapshots
                        WHERE camera_id NOT LIKE 'SIM_%%'
                        ORDER BY camera_id, timestamp DESC
                    ) latest_per_camera
                """)
                live_snap = cur.fetchone()
                total_queue  = int((live_snap or {}).get("queue_count") or 0)
                live_avg_sec = float((live_snap or {}).get("avg_dwell_sec") or 0)

                # 2. Median service time from recent service_events (same as dashboard)
                cur.execute("""
                    SELECT COALESCE(
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_dwell_sec / 60.0),
                        2.0
                    ) AS service_min
                    FROM service_events
                    WHERE timestamp >= NOW() - INTERVAL '7 days'
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND total_dwell_sec > 0
                """)
                svc = cur.fetchone()
                service_min = max(0.5, float((svc or {}).get("service_min") or 2.0))

                # 3. Current open lanes
                cur.execute("""
                    SELECT COUNT(DISTINCT lane_id) AS active_lanes
                    FROM service_events
                    WHERE timestamp >= NOW() - INTERVAL '4 hours'
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND lane_id IS NOT NULL
                """)
                lane_row = cur.fetchone()
                current_lanes = max(int((lane_row or {}).get("active_lanes") or 1), 1)

                # 4. Predicted future arrivals + precomputed lane-scenario waits.
                #    DISTINCT ON picks the best prediction per slot (same as dashboard).
                cur.execute("""
                    SELECT DISTINCT ON (prediction_for)
                        prediction_for,
                        COALESCE(ensemble_yhat, 0) AS arrivals,
                        wait_1lane_15m,
                        wait_2lane_15m,
                        wait_3lane_15m
                    FROM queue_predictions
                    WHERE prediction_for >= NOW()
                      AND prediction_for <= NOW() + INTERVAL '20 minutes'
                    ORDER BY prediction_for ASC, predicted_at DESC
                """)
                pred_rows = cur.fetchall()

        # ── Queue simulation (mirrors dashboard's _compute_waits_for_lanes) ──────
        # Capacity: how many customers each lane serves per BUCKET_MIN window
        served_per_bucket = current_lanes * (BUCKET_MIN / service_min)

        # Use backlog (people waiting, not yet being served) as the dashboard does:
        # queue_count - open_lanes = people waiting; those being served don't add wait.
        queue = max(0.0, float(total_queue) - float(current_lanes))
        now_utc = datetime.now(timezone.utc)
        sim_results: dict[int, float] = {}

        for row in pred_rows:
            pf = row["prediction_for"]
            if pf.tzinfo is None:
                pf = pf.replace(tzinfo=timezone.utc)
            slot_min = (pf - now_utc).total_seconds() / 60
            if slot_min < 0:
                continue

            arrivals = float(row["arrivals"] or 0)
            queue = max(0.0, queue + arrivals - served_per_bucket)
            wait = round((queue / current_lanes) * service_min, 1) if current_lanes > 0 and queue > 0 else 0.0

            for target in (5, 10, 15):
                if slot_min >= target and target not in sim_results:
                    sim_results[target] = wait

        # Fallback: no DB predictions — decay from current state
        if not sim_results and total_queue > 0:
            for target in (5, 10, 15):
                remaining = max(0.0, total_queue - target * current_lanes / service_min)
                sim_results[target] = round((remaining / current_lanes) * service_min, 1) if remaining > 0 else 0.0

        wait_now = round(live_avg_sec / 60, 1) if live_avg_sec > 0 else 0.0
        wait_5   = sim_results.get(5)
        wait_10  = sim_results.get(10)
        wait_15  = sim_results.get(15)

        # Lane scenarios from precomputed ensemble columns (15-min horizon per N lanes).
        # 4-lane value scaled from 3-lane since ensemble only computes up to 3.
        base = pred_rows[0] if pred_rows else None
        w1 = round(float(base["wait_1lane_15m"] or 0), 1) if base else 0.0
        w2 = round(float(base["wait_2lane_15m"] or 0), 1) if base else 0.0
        w3 = round(float(base["wait_3lane_15m"] or 0), 1) if base else 0.0
        w4 = round(w3 * 0.75, 1) if w3 > 0 else 0.0
        lane_waits = {1: w1, 2: w2, 3: w3, 4: w4}

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
            "date":            display_date,
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