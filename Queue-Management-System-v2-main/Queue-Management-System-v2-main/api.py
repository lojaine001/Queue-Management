"""
api.py — FastAPI backend for the IQMS Manager Mobile App.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import csv
import io
import os
import sys
from zoneinfo import ZoneInfo
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Every other script in this codebase (dashboard.py, ensemble_predict.py,
# run_scheduler.py, push_alert_checker.py, ...) loads .env this way --
# this file never did, so anything set only in .env (not a real OS-level
# env var) was silently invisible to it. Caught via VAPID_PUBLIC_KEY
# returning "not configured" despite being set in .env.
load_dotenv(find_dotenv(usecwd=True))

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from prediction.forecast_state import finite, freshness, lane_scenarios, scenario_for, wait_at_horizon

import camera_snapshot

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


def kpi_wait(cur, start_date, end_date, bucket="hour"):
    """Average/max checkout wait (minutes), from service_events.
    start_date inclusive, end_date exclusive (both dates, store TZ).
    bucket='hour' (used for a single day) or 'day' (used for week/month)."""
    key = "by_hour" if bucket == "hour" else "by_day"
    if start_date is None:
        return {"avg_wait_min": None, "max_wait_min": None, "wait_source": "service_events", key: []}
    trunc_unit = "hour" if bucket == "hour" else "day"
    fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d"
    label_key = "hour" if bucket == "hour" else "date"

    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket,
               ROUND(AVG(total_dwell_sec)::numeric / 60.0, 1) AS avg_wait_min,
               ROUND(MAX(total_dwell_sec)::numeric / 60.0, 1) AS max_wait_min
        FROM service_events
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1 ASC
    """, (trunc_unit, STORE_TZ, CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    rows = cur.fetchall()

    cur.execute("""
        SELECT ROUND(AVG(total_dwell_sec)::numeric / 60.0, 1) AS avg_wait_min,
               ROUND(MAX(total_dwell_sec)::numeric / 60.0, 1) AS max_wait_min
        FROM service_events
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
    """, (CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    overall = cur.fetchone() or {}

    return {
        "avg_wait_min": float(overall["avg_wait_min"]) if overall.get("avg_wait_min") is not None else None,
        "max_wait_min": float(overall["max_wait_min"]) if overall.get("max_wait_min") is not None else None,
        "wait_source": "service_events",
        key: [
            {
                label_key: r["bucket"].strftime(fmt),
                "avg_wait_min": float(r["avg_wait_min"]) if r["avg_wait_min"] is not None else None,
                "max_wait_min": float(r["max_wait_min"]) if r["max_wait_min"] is not None else None,
            }
            for r in rows
        ],
    }


def kpi_queue(cur, start_date, end_date, bucket="hour"):
    """Average/peak people waiting at checkout, from queue_state_snapshots.
    Same start/end/bucket contract as kpi_wait."""
    key = "by_hour" if bucket == "hour" else "by_day"
    if start_date is None:
        return {"avg_waiting": None, "peak_waiting": None, "peak_time": None, key: []}
    trunc_unit = "hour" if bucket == "hour" else "day"
    fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d"
    label_key = "hour" if bucket == "hour" else "date"

    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket,
               ROUND(AVG(queue_count)::numeric, 1) AS avg_waiting,
               MAX(queue_count) AS max_waiting
        FROM queue_state_snapshots
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1 ASC
    """, (trunc_unit, STORE_TZ, CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    rows = cur.fetchall()

    cur.execute("""
        SELECT ROUND(AVG(queue_count)::numeric, 1) AS avg_waiting, MAX(queue_count) AS peak_waiting
        FROM queue_state_snapshots
        WHERE camera_id = %s
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
    """, (CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date))
    overall = cur.fetchone() or {}

    peak_time = None
    if overall.get("peak_waiting") is not None:
        cur.execute("""
            SELECT timestamp FROM queue_state_snapshots
            WHERE camera_id = %s
              AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
              AND queue_count = %s
            ORDER BY timestamp ASC LIMIT 1
        """, (CHECKOUT_CAM_ID, STORE_TZ, start_date, STORE_TZ, end_date, overall["peak_waiting"]))
        peak_row = cur.fetchone()
        if peak_row:
            peak_fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d %H:%M"
            peak_time = peak_row["timestamp"].astimezone(ZoneInfo(STORE_TZ)).strftime(peak_fmt)

    return {
        "avg_waiting": float(overall["avg_waiting"]) if overall.get("avg_waiting") is not None else None,
        "peak_waiting": int(overall["peak_waiting"]) if overall.get("peak_waiting") is not None else None,
        "peak_time": peak_time,
        key: [
            {
                label_key: r["bucket"].strftime(fmt),
                "avg_waiting": float(r["avg_waiting"]) if r["avg_waiting"] is not None else None,
                "max_waiting": int(r["max_waiting"]) if r["max_waiting"] is not None else None,
            }
            for r in rows
        ],
    }


# ── Models ────────────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    response: str
    lane_id: Optional[str] = None


class SetLanesRequest(BaseModel):
    lanes: int


class PushKeys(BaseModel):
    p256dh: str
    auth: str


class PushSubscribeRequest(BaseModel):
    endpoint: str
    keys: PushKeys
    # The device's own alert setting (its threshold slider + selected
    # horizon), sent along with the subscription so a server-side push can
    # fire at exactly the same point this device's in-page alert would --
    # not a separate, possibly-disagreeing threshold.
    threshold_min: float
    horizon_min: int


class PushUnsubscribeRequest(BaseModel):
    endpoint: str


def _ensure_push_table():
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS push_subscriptions (
                    endpoint      TEXT PRIMARY KEY,
                    p256dh        TEXT NOT NULL,
                    auth          TEXT NOT NULL,
                    threshold_min DOUBLE PRECISION NOT NULL,
                    horizon_min   INTEGER NOT NULL,
                    -- Server-side rising-edge tracker, one per subscription --
                    -- same "fire only on crossing into alert, not every poll"
                    -- rule the frontend's own alert effect already follows.
                    was_over      BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        conn.commit()


try:
    _ensure_push_table()
except Exception as exc:
    # Don't crash the whole API on startup if the DB isn't reachable yet --
    # every other endpoint needs it too and already fails per-request
    # instead, so match that rather than taking the app down entirely.
    print(f"[push] Could not ensure push_subscriptions table exists: {exc}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/vapid-public-key")
def vapid_public_key():
    key = os.getenv("VAPID_PUBLIC_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="VAPID_PUBLIC_KEY not configured on the server")
    return {"public_key": key}


@app.post("/push-subscribe")
def push_subscribe(body: PushSubscribeRequest):
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO push_subscriptions
                        (endpoint, p256dh, auth, threshold_min, horizon_min, was_over, updated_at)
                    VALUES (%s, %s, %s, %s, %s, FALSE, NOW())
                    ON CONFLICT (endpoint) DO UPDATE SET
                        p256dh        = EXCLUDED.p256dh,
                        auth          = EXCLUDED.auth,
                        -- Changing the threshold or horizon changes what
                        -- "already alerted" even means, so reset the
                        -- rising-edge tracker whenever either changes --
                        -- same reasoning as the frontend's own
                        -- horizon-switch resync (see LiveScreen.jsx).
                        was_over      = CASE
                            WHEN push_subscriptions.threshold_min IS DISTINCT FROM EXCLUDED.threshold_min
                              OR push_subscriptions.horizon_min   IS DISTINCT FROM EXCLUDED.horizon_min
                            THEN FALSE
                            ELSE push_subscriptions.was_over
                        END,
                        threshold_min = EXCLUDED.threshold_min,
                        horizon_min   = EXCLUDED.horizon_min,
                        updated_at    = NOW()
                """, (body.endpoint, body.keys.p256dh, body.keys.auth, body.threshold_min, body.horizon_min))
            conn.commit()
        return {"status": "subscribed"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/push-unsubscribe")
def push_unsubscribe(body: PushUnsubscribeRequest):
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM push_subscriptions WHERE endpoint = %s", (body.endpoint,))
            conn.commit()
        return {"status": "unsubscribed"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


def _forecast_state():
    conn = _conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT d.*, f.revision AS current_settings_revision FROM dashboard_state d LEFT JOIN forecast_settings f ON f.id=d.id WHERE d.id=1")
            return cur.fetchone() or {}
    finally:
        conn.close()


@app.get("/forecast")
def forecast():
    """Return the lane simulations published by the background worker."""
    try:
        state = _forecast_state()
        scenario = scenario_for(state)
        def value(key):
            number = finite(scenario.get(key) if scenario else state.get(key))
            return round(number, 1) if number is not None else None
        return {
            "wait_now_min": value("wait_0m"), "wait_5_min": value("wait_5m"),
            "wait_10_min": value("wait_10m"), "wait_15_min": value("wait_15m"),
            "current_lanes": int(state.get("open_lanes") or 1),
            "lane_scenarios": lane_scenarios(state),
            "queue_now": state.get("queue_now"),
            "simulation_available": bool(scenario.get("slots")),
            **freshness(state),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/forecast/wait")
def forecast_wait(minutes: float = 0):
    """Read a horizon relative to the shared forecast's calculation time."""
    try:
        return wait_at_horizon(_forecast_state(), minutes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
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

                # Hourly entries — computed fresh per requested day, not from
                # dashboard_state, which only ever holds today's cache and
                # would be wrong for past dates.
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
                wait_stats = kpi_wait(cur, ref_date, ref_date + timedelta(days=1)) if ref_date else kpi_wait(cur, None, None)
                queue_stats = kpi_queue(cur, ref_date, ref_date + timedelta(days=1)) if ref_date else kpi_queue(cur, None, None)

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
            "entries_by_hour":     hourly_entries, "wait_stats": wait_stats, "queue_stats": queue_stats,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _shared_forecast_chart(horizon_minutes):
    state = _forecast_state()
    slots = scenario_for(state).get("slots", [])
    now = datetime.now(timezone.utc)
    end = now + timedelta(minutes=horizon_minutes)
    result = []
    for row in slots:
        stamp = datetime.fromisoformat(row["prediction_for"])
        if now <= stamp <= end:
            result.append({"time": stamp.astimezone(ZoneInfo(STORE_TZ)).strftime("%H:%M"),
                           "prediction_for": row["prediction_for"],
                           "arrivals": round(row["arrivals"], 1) if row.get("arrivals") is not None else None,
                           "wait_min": round(row["wait_min"], 1) if row.get("wait_min") is not None else None})
    return {"slots": result, "current_lanes": state.get("open_lanes"), **freshness(state)}


@app.get("/forecast-chart")
def forecast_chart():
    """The current lane's dashboard simulation, over the next 60 minutes."""
    try:
        return _shared_forecast_chart(60)
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
    try:
        return _shared_forecast_chart(180)
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
def snapshot_checkout(request: Request, format: str = None):
    if format == "json":
        # Legacy JSON form, kept for IQMSManager compatibility until it is
        # updated to the new binary endpoint too.
        import base64
        p = SNAP_DIR / "latest_checkout.jpg"
        if not p.exists():
            return {"image": None}
        with open(str(p), "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return {"image": f"data:image/jpeg;base64,{data}"}

    result = camera_snapshot.get_snapshot()
    if result["bytes"] is None:
        raise HTTPException(status_code=503, detail="Checkout camera unavailable")
    if request.headers.get("if-none-match") == result["etag"]:
        return Response(status_code=304, headers={"ETag": result["etag"]})
    return Response(
        content=result["bytes"],
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache",
            "ETag": result["etag"],
            "X-Snapshot-Age": str(round(result["age_seconds"], 1)),
        },
    )


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
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM dashboard_state WHERE id = 1 FOR UPDATE")
                state = cur.fetchone() or {}
                scenario = scenario_for(state, body.lanes)
                if not scenario or freshness(state)["stale"]:
                    raise HTTPException(status_code=409, detail="Wait for a fresh background forecast before changing lanes.")
                cur.execute(
                    """UPDATE dashboard_state
                       SET open_lanes = %s, wait_0m = %s, wait_5m = %s, wait_10m = %s, wait_15m = %s
                       WHERE id = 1""",
                    (body.lanes, finite(scenario.get("wait_0m")), finite(scenario.get("wait_5m")),
                     finite(scenario.get("wait_10m")), finite(scenario.get("wait_15m"))),
                )
            conn.commit()
        # Changing the lane selection does not make the simulation's inputs newer.
        return {"status": "ok", "lanes": body.lanes, **freshness(state)}
    except HTTPException:
        raise
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


@app.get("/statistics/summary")
def statistics_summary():
    """Background demographic/traffic snapshot, with its own freshness timestamp."""
    conn=_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT * FROM dashboard_statistics WHERE id=1')
            row=dict(cur.fetchone() or {})
        if not row:return {"available":False,"stale":True}
        return {"available":True,**row,**freshness({"updated_at":row["updated_at"]})}
    finally:conn.close()

PERIOD_BUCKET = {"day": "hour", "week": "day", "month": "day"}

ALL_KPIS = ["customers", "avg_wait", "max_wait", "avg_waiting", "peak_waiting",
            "lanes_used", "alert_minutes", "trolley", "store_basket", "peak_hour"]

KPI_LABELS = {
    "fr": {
        "customers": "Clients total", "avg_wait": "Temps d'attente moyen (min)",
        "max_wait": "Temps d'attente max (min)", "avg_waiting": "Personnes en attente (moy.)",
        "peak_waiting": "Pic de personnes en attente", "lanes_used": "Files utilisees",
        "alert_minutes": "Temps en alerte (min)", "trolley": "Chariots", "store_basket": "Paniers",
        "peak_hour": "Heure de pointe",
    },
    "en": {
        "customers": "Total customers", "avg_wait": "Avg checkout wait (min)",
        "max_wait": "Max checkout wait (min)", "avg_waiting": "Avg people waiting",
        "peak_waiting": "Peak people waiting", "lanes_used": "Lanes used",
        "alert_minutes": "Time in alert (min)", "trolley": "Trolleys", "store_basket": "Baskets",
        "peak_hour": "Peak hour",
    },
}


def _period_bounds(period: str, date_str: str):
    try:
        ref = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    if period == "day":
        return ref, ref + timedelta(days=1)
    elif period == "week":
        monday = ref - timedelta(days=ref.weekday())
        return monday, monday + timedelta(days=7)
    elif period == "month":
        first = ref.replace(day=1)
        next_first = (first.replace(year=first.year + 1, month=1)
                      if first.month == 12 else first.replace(month=first.month + 1))
        return first, next_first
    raise HTTPException(status_code=400, detail="period must be day, week, or month")


def _export_rows(cur, start_date, end_date, bucket):
    """Returns (rows, total_row). rows: one per hour (day period) or per day
    (week/month period). total_row: sums for counts, the true overall
    avg/max from kpi_wait/kpi_queue (not an average-of-averages), and a
    true distinct-lane count over the whole range (not a sum, which would
    double-count a lane reused across multiple days)."""
    trunc_unit = "hour" if bucket == "hour" else "day"
    fmt = "%H:%M" if bucket == "hour" else "%Y-%m-%d"

    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket,
               COUNT(*) AS customers,
               COUNT(*) FILTER (WHERE equipment_type = 'trolley') AS trolley,
               COUNT(*) FILTER (WHERE equipment_type = 'store_basket') AS store_basket
        FROM entrance_events
        WHERE camera_id NOT LIKE 'SIM_%%' AND dwell_seconds >= 10
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1
    """, (trunc_unit, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
    entrance_by = {r["bucket"].strftime(fmt): r for r in cur.fetchall()}

    wait = kpi_wait(cur, start_date, end_date, bucket)
    wait_key = "by_hour" if bucket == "hour" else "by_day"
    wait_by = {(r.get("hour") or r.get("date")): r for r in wait[wait_key]}

    queue = kpi_queue(cur, start_date, end_date, bucket)
    queue_by = {(r.get("hour") or r.get("date")): r for r in queue[wait_key]}

    cur.execute("""
        SELECT DATE_TRUNC(%s, timestamp AT TIME ZONE %s) AS bucket, COUNT(DISTINCT lane_id) AS lanes_used
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1
    """, (trunc_unit, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
    lanes_by = {r["bucket"].strftime(fmt): int(r["lanes_used"]) for r in cur.fetchall()}

    cur.execute("""
        SELECT DATE_TRUNC(%s, prediction_for AT TIME ZONE %s) AS bucket, COUNT(*) AS alert_slots
        FROM queue_predictions
        WHERE status = 'ALERT'
          AND prediction_for AT TIME ZONE %s >= %s AND prediction_for AT TIME ZONE %s < %s
        GROUP BY 1 ORDER BY 1
    """, (trunc_unit, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
    alert_by = {r["bucket"].strftime(fmt): int(r["alert_slots"]) * BUCKET_MIN for r in cur.fetchall()}

    peak_hour_by = {}
    if bucket == "day":
        cur.execute("""
            SELECT DATE_TRUNC('day', timestamp AT TIME ZONE %s) AS day,
                   DATE_TRUNC('hour', timestamp AT TIME ZONE %s) AS hour,
                   COUNT(*) AS cnt
            FROM entrance_events
            WHERE camera_id NOT LIKE 'SIM_%%' AND dwell_seconds >= 10
              AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
            GROUP BY 1, 2
        """, (STORE_TZ, STORE_TZ, STORE_TZ, start_date, STORE_TZ, end_date))
        by_day_hours = {}
        for r in cur.fetchall():
            d = r["day"].strftime("%Y-%m-%d")
            by_day_hours.setdefault(d, []).append((r["hour"], int(r["cnt"])))
        for d, hrs in by_day_hours.items():
            peak_hour_by[d] = max(hrs, key=lambda x: x[1])[0].strftime("%H:%M")

    rows = []
    if bucket == "hour":
        for h in range(24):
            label = f"{h:02d}:00"
            e = entrance_by.get(label)
            w = wait_by.get(label, {})
            q = queue_by.get(label, {})
            rows.append({
                "label": label, "customers": int(e["customers"]) if e else 0,
                "avg_wait": w.get("avg_wait_min"), "max_wait": w.get("max_wait_min"),
                "avg_waiting": q.get("avg_waiting"), "peak_waiting": q.get("max_waiting"),
                "lanes_used": lanes_by.get(label, 0), "alert_minutes": alert_by.get(label, 0),
                "trolley": int(e["trolley"]) if e else 0, "store_basket": int(e["store_basket"]) if e else 0,
                "peak_hour": None,
            })
    else:
        d = start_date
        while d < end_date:
            label = d.strftime("%Y-%m-%d")
            e = entrance_by.get(label)
            w = wait_by.get(label, {})
            q = queue_by.get(label, {})
            rows.append({
                "label": label, "customers": int(e["customers"]) if e else 0,
                "avg_wait": w.get("avg_wait_min"), "max_wait": w.get("max_wait_min"),
                "avg_waiting": q.get("avg_waiting"), "peak_waiting": q.get("max_waiting"),
                "lanes_used": lanes_by.get(label, 0), "alert_minutes": alert_by.get(label, 0),
                "trolley": int(e["trolley"]) if e else 0, "store_basket": int(e["store_basket"]) if e else 0,
                "peak_hour": peak_hour_by.get(label),
            })
            d += timedelta(days=1)

    cur.execute("""
        SELECT COUNT(DISTINCT lane_id) AS lanes_used
        FROM service_events
        WHERE camera_id NOT LIKE 'SIM_%%'
          AND timestamp AT TIME ZONE %s >= %s AND timestamp AT TIME ZONE %s < %s
    """, (STORE_TZ, start_date, STORE_TZ, end_date))
    total_lanes_row = cur.fetchone()
    total_lanes = int(total_lanes_row["lanes_used"]) if total_lanes_row and total_lanes_row["lanes_used"] else 0

    total_row = {
        "label": "TOTAL",
        "customers": sum(r["customers"] for r in rows),
        "avg_wait": wait.get("avg_wait_min"),
        "max_wait": wait.get("max_wait_min"),
        "avg_waiting": queue.get("avg_waiting"),
        "peak_waiting": queue.get("peak_waiting"),
        "lanes_used": total_lanes,
        "alert_minutes": sum(r["alert_minutes"] for r in rows),
        "trolley": sum(r["trolley"] for r in rows),
        "store_basket": sum(r["store_basket"] for r in rows),
        "peak_hour": None,
    }

    return rows, total_row


@app.get("/export/preview")
def export_preview(period: str, date: str, kpis: Optional[str] = None, lang: str = "fr"):
    if period not in PERIOD_BUCKET:
        raise HTTPException(status_code=400, detail="period must be day, week, or month")
    start_date, end_date = _period_bounds(period, date)
    bucket = PERIOD_BUCKET[period]
    selected = kpis.split(",") if kpis else ALL_KPIS
    unknown = [k for k in selected if k not in ALL_KPIS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown KPI(s): {unknown}")

    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            rows, total_row = _export_rows(cur, start_date, end_date, bucket)

    labels = KPI_LABELS.get(lang, KPI_LABELS["fr"])
    return {
        "period": period, "bucket": bucket, "kpis": selected,
        "labels": {k: labels[k] for k in selected},
        "rows": [{"label": r["label"], **{k: r[k] for k in selected}} for r in rows[:10]],
        "total": {"label": total_row["label"], **{k: total_row[k] for k in selected}},
        "row_count": len(rows),
    }


@app.get("/export")
def export_csv(period: str, date: str, kpis: Optional[str] = None, format: str = "csv", lang: str = "fr"):
    if format != "csv":
        raise HTTPException(status_code=400, detail="Only format=csv is supported currently")
    if period not in PERIOD_BUCKET:
        raise HTTPException(status_code=400, detail="period must be day, week, or month")
    start_date, end_date = _period_bounds(period, date)
    bucket = PERIOD_BUCKET[period]
    selected = kpis.split(",") if kpis else ALL_KPIS
    unknown = [k for k in selected if k not in ALL_KPIS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown KPI(s): {unknown}")

    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            rows, total_row = _export_rows(cur, start_date, end_date, bucket)

    labels = KPI_LABELS.get(lang, KPI_LABELS["fr"])
    sep = ";" if lang == "fr" else ","
    row_label = ("Heure" if bucket == "hour" else "Date") if lang == "fr" else ("Hour" if bucket == "hour" else "Date")

    def fmt_num(v):
        if v is None:
            return ""
        s = f"{v}"
        return s.replace(".", ",") if lang == "fr" else s

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=sep)
    writer.writerow([row_label] + [labels[k] for k in selected])
    for r in rows:
        writer.writerow([r["label"]] + [fmt_num(r[k]) for k in selected])
    writer.writerow([total_row["label"]] + [fmt_num(total_row[k]) for k in selected])

    csv_bytes = ("﻿" + buf.getvalue()).encode("utf-8")

    if period == "day":
        filename = f"IQMS_{date}.csv"
    elif period == "week":
        iso_year, iso_week, _ = start_date.isocalendar()
        filename = f"IQMS_semaine_{iso_year}-W{iso_week:02d}.csv"
    else:
        filename = f"IQMS_mois_{start_date.strftime('%Y-%m')}.csv"

    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
