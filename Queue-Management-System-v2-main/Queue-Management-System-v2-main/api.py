"""
api.py — FastAPI backend for the IQMS Manager Mobile App.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

SNAP_DIR = Path(__file__).resolve().parent / "snapshots"

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
CHECKOUT_CAM_ID = os.getenv("CHECKOUT_CAM_ID", "Bosch_Camera_exit")
BUCKET_MIN = int(os.getenv("BUCKET_MINUTES", 3))


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


class SetLanesRequest(BaseModel):
    lanes: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/live-lanes")
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
            depth = int(lane_counts.get(str(i), 0))
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

        # Per-lane scenario waits at +10 min horizon.
        # lane{n}_wait_10m is not stored in dashboard_state, so we derive it
        # proportionally from wait_10m: demand is ~constant at this horizon,
        # so wait scales linearly with 1/lanes.
        scenarios = []
        for n in range(1, 5):
            if wait_10 is not None and current_lanes > 0:
                estimated = round(wait_10 * current_lanes / n, 1)
            else:
                estimated = 0.0
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
            "queue_now":      int(state["queue_now"]) if state["queue_now"] is not None else None,
            "updated_at":     state["updated_at"].isoformat() if state["updated_at"] else None,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/day-recap")
def day_recap(date: Optional[str] = None):
    """
    Returns the summary for a given day (?date=YYYY-MM-DD), or falls back
    to the most recent day with data when no date is given.
    """
    try:
        requested_date = None
        if date:
            try:
                requested_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format")

        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if requested_date:
                    ref_date = requested_date
                else:
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
                      AND dwell_seconds >= 10
                """)
                total = int((cur.fetchone() or {}).get("total") or 0)

                # Yesterday's total, for the vs-yesterday comparison
                yesterday_date = (ref_date - timedelta(days=1)) if ref_date else None
                yesterday_total = None
                if yesterday_date:
                    cur.execute("""
                        SELECT COUNT(*) AS total
                        FROM entrance_events
                        WHERE DATE(timestamp) = %s
                          AND camera_id NOT LIKE 'SIM_%%'
                          AND dwell_seconds >= 10
                    """, (yesterday_date,))
                    yesterday_total = int((cur.fetchone() or {}).get("total") or 0)

                # 7-day trend ending on ref_date, for the Clients Total sparkline
                trend_rows = []
                if ref_date:
                    cur.execute("""
                        SELECT DATE(timestamp) AS day, COUNT(*) AS cnt
                        FROM entrance_events
                        WHERE DATE(timestamp) BETWEEN %s AND %s
                          AND camera_id NOT LIKE 'SIM_%%'
                          AND dwell_seconds >= 10
                        GROUP BY 1 ORDER BY 1 ASC
                    """, (ref_date - timedelta(days=6), ref_date))
                    trend_rows = cur.fetchall()

                cur.execute(f"""
                    SELECT DATE_TRUNC('hour', timestamp AT TIME ZONE 'Europe/Paris') AS hour, COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND dwell_seconds >= 10
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

                # Lanes used today
                cur.execute(f"""
                    SELECT COUNT(DISTINCT lane_id) AS lanes_count
                    FROM service_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                """)
                lanes_row    = cur.fetchone()
                lanes_today  = int(lanes_row["lanes_count"]) if lanes_row and lanes_row["lanes_count"] else 0

                cur.execute(f"""
                    SELECT lane_id, COUNT(*) AS cnt
                    FROM service_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                    GROUP BY lane_id ORDER BY cnt DESC LIMIT 1
                """)
                busiest_row  = cur.fetchone()
                busiest_lane = busiest_row["lane_id"] if busiest_row else None

                # Alert minutes today (from ensemble predictions)
                alert_date_filter = f"DATE(prediction_for) = '{ref_date}'" if ref_date else "DATE(prediction_for) = CURRENT_DATE"
                cur.execute(f"""
                    SELECT COUNT(*) AS alert_slots
                    FROM queue_predictions
                    WHERE {alert_date_filter}
                      AND status = 'ALERT'
                """)
                alert_row     = cur.fetchone()
                alert_minutes = int((alert_row["alert_slots"] or 0) if alert_row else 0) * BUCKET_MIN

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

                # Gender/age demographics — computed fresh for the requested
                # day (not read from dashboard_state, which only ever holds
                # today's cache and would be wrong for past dates).
                cur.execute(f"""
                    SELECT gender, COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND dwell_seconds >= 10
                      AND gender IS NOT NULL AND gender != 'unknown'
                    GROUP BY gender
                """)
                gender_rows = cur.fetchall()

                cur.execute(f"""
                    SELECT
                        CASE
                            WHEN age_estimate < 30 THEN '18-30'
                            WHEN age_estimate < 50 THEN '30-50'
                            ELSE '50+'
                        END AS age_group,
                        COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND dwell_seconds >= 10
                      AND age_estimate IS NOT NULL
                    GROUP BY 1
                    ORDER BY MIN(age_estimate)
                """)
                age_rows = cur.fetchall()

                cur.execute(f"""
                    SELECT ROUND(AVG(age_estimate)::numeric, 1) AS avg_age
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND dwell_seconds >= 10
                      AND age_estimate IS NOT NULL
                """)
                avg_age_row = cur.fetchone()

                # Hourly entries — also computed fresh per requested day,
                # same reason as demographics above.
                cur.execute(f"""
                    SELECT DATE_TRUNC('hour', timestamp AT TIME ZONE 'Europe/Paris') AS hour,
                           COUNT(*) AS cnt
                    FROM entrance_events
                    WHERE {date_filter}
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND dwell_seconds >= 10
                    GROUP BY 1 ORDER BY 1 ASC
                """)
                hourly_rows = cur.fetchall()

        trend_by_day = {r["day"]: int(r["cnt"]) for r in trend_rows}
        trend_7d = []
        if ref_date:
            for i in range(6, -1, -1):
                d = ref_date - timedelta(days=i)
                trend_7d.append({"date": d.strftime("%Y-%m-%d"), "count": trend_by_day.get(d, 0)})

        denom = total or 1
        equipment = []
        order  = ["trolley", "store_basket"]
        colors = {"trolley": "#06b6d4", "store_basket": "#a855f7"}
        labels = {"trolley": "Trolley", "store_basket": "Store basket"}
        for key in order:
            row = next((r for r in equip_rows if r["equipment_type"] == key), None)
            count = int(row["cnt"]) if row else 0
            equipment.append({
                "type":    key,
                "label":   labels[key],
                "count":   count,
                "percent": round(count / denom * 100),
                "color":   colors[key],
            })

        gender_total = sum(int(r["cnt"]) for r in gender_rows) or 1
        gender_colors = {"female": "#ec4899", "male": "#3b82f6"}
        gender_labels = {"female": "Femme", "male": "Homme"}
        demographics_gender = [
            {
                "key":     r["gender"],
                "label":   gender_labels.get(r["gender"], str(r["gender"]).capitalize()),
                "count":   int(r["cnt"]),
                "percent": round(int(r["cnt"]) / gender_total * 100),
                "color":   gender_colors.get(r["gender"], "#94a3b8"),
            }
            for r in gender_rows
        ]

        age_colors = {"18-30": "#22d3ee", "30-50": "#a78bfa", "50+": "#fb923c"}
        demographics_age = [
            {
                "group": r["age_group"],
                "count": int(r["cnt"]),
                "color": age_colors.get(r["age_group"], "#94a3b8"),
            }
            for r in age_rows
        ]

        avg_age = float(avg_age_row["avg_age"]) if avg_age_row and avg_age_row["avg_age"] else None

        peak_hourly_cnt = max((int(r["cnt"]) for r in hourly_rows), default=0)
        hourly_entries = [
            {
                "hour":    r["hour"].strftime("%H:00"),
                "count":   int(r["cnt"]),
                "is_peak": int(r["cnt"]) == peak_hourly_cnt and peak_hourly_cnt > 0,
            }
            for r in hourly_rows
        ]

        vs_yesterday_pct = (
            round((total - yesterday_total) / yesterday_total * 100)
            if yesterday_total else None
        )
        peak_pct_of_total = round(peak_count / total * 100) if total else None

        return {
            "date":                display_date,
            "total_customers":     total,
            "vs_yesterday_pct":    vs_yesterday_pct,
            "trend_7d":            trend_7d,
            "avg_wait_min":        avg_wait_min,
            "peak_hour":           peak_hour,
            "peak_hour_end":       peak_end,
            "peak_count":          peak_count,
            "peak_pct_of_total":   peak_pct_of_total,
            "equipment":           equipment,
            "lanes_today":         lanes_today,
            "busiest_lane":        busiest_lane,
            "alert_minutes":       alert_minutes,
            "demographics_gender": demographics_gender,
            "demographics_age":    demographics_age,
            "avg_age":             avg_age,
            "entries_by_hour":     hourly_entries,
        }

    except HTTPException:
        raise
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


@app.get("/day-wait-chart")
def day_wait_chart(date: Optional[str] = None):
    """
    Full-day history of predicted arrivals/wait for the Statistique page's
    Temps d'attente chart. Unlike /forecast-chart (always the next 60 min
    forward from now), this covers a whole day - past or current - at
    native prediction resolution. For today, it naturally tapers off
    wherever the ensemble job's predictions currently stop (it doesn't
    force the line to stop exactly at "now" or fabricate future points).
    """
    try:
        if date:
            try:
                ref_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format")
        else:
            ref_date = datetime.now().date()

        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT DISTINCT ON (prediction_for)
                        prediction_for,
                        COALESCE(ensemble_yhat, 0)    AS arrivals,
                        COALESCE(est_wait_minutes, 0) AS wait_min
                    FROM queue_predictions
                    WHERE DATE(prediction_for) = %s
                    ORDER BY prediction_for ASC, predicted_at DESC
                """, (ref_date,))
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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/forecast-chart-3h")
def forecast_chart_3h():
    """Returns 3-hour time series of predicted arrivals and wait for the app chart."""
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
                      AND prediction_for <= NOW() + INTERVAL '3 hours'
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


@app.get("/forecast-chart-12h")
def forecast_chart_12h():
    """Returns last 6 hours of actual entrance counts (3-min buckets) for the app chart."""
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        time_bucket('3 minutes', timestamp) AS slot,
                        COUNT(*) AS entries
                    FROM entrance_events
                    WHERE timestamp >= NOW() - INTERVAL '6 hours'
                      AND timestamp <= NOW()
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND dwell_seconds >= 10
                    GROUP BY 1
                    ORDER BY 1 ASC
                """)
                rows = cur.fetchall()
        return {
            "slots": [
                {
                    "time":    row["slot"].strftime("%H:%M"),
                    "entries": int(row["entries"]),
                }
                for row in rows
            ]
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/forecast-chart-2d")
def forecast_chart_2d():
    """Returns last 2 days of actual entrance counts (3-min buckets) for the app chart."""
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        time_bucket('3 minutes', timestamp) AS slot,
                        COUNT(*) AS entries
                    FROM entrance_events
                    WHERE timestamp >= NOW() - INTERVAL '2 days'
                      AND timestamp <= NOW()
                      AND camera_id NOT LIKE 'SIM_%%'
                      AND dwell_seconds >= 10
                    GROUP BY 1
                    ORDER BY 1 ASC
                """)
                rows = cur.fetchall()
        return {
            "slots": [
                {
                    "time":    row["slot"].strftime("%d/%m %H:%M"),
                    "entries": int(row["entries"]),
                }
                for row in rows
            ]
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/snapshot/checkout")
def snapshot_checkout():
    import base64
    p = SNAP_DIR / "latest_checkout.jpg"
    if not p.exists():
        return {"image": None}
    with open(str(p), "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"image": f"data:image/jpeg;base64,{data}"}


@app.get("/snapshot/entrance")
def snapshot_entrance():
    import base64
    p = SNAP_DIR / "latest_entrance.jpg"
    if not p.exists():
        return {"image": None}
    with open(str(p), "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"image": f"data:image/jpeg;base64,{data}"}


@app.post("/set-lanes")
def set_lanes(body: SetLanesRequest):
    if not (1 <= body.lanes <= 4):
        raise HTTPException(status_code=400, detail="lanes must be between 1 and 4")
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE dashboard_state SET open_lanes = %s, updated_at = NOW() WHERE id = 1",
                    (body.lanes,),
                )
            conn.commit()
        return {"status": "ok", "lanes": body.lanes}
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
