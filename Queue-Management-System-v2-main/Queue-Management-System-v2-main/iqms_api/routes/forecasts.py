"""Published forecasts: current waits per lane scenario, horizons and charts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, HTTPException

from prediction.forecast_state import finite, freshness, lane_scenarios, scenario_for, wait_at_horizon

from ..db import _conn, _http_error
from ..repositories import _forecast_state
from ..settings import STORE_TZ

router = APIRouter()


@router.get("/forecast")
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
        raise _http_error(exc)


@router.get("/forecast/wait")
def forecast_wait(minutes: float = 0):
    """Read a horizon relative to the shared forecast's calculation time."""
    try:
        return wait_at_horizon(_forecast_state(), minutes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise _http_error(exc)


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


@router.get("/forecast-chart")
def forecast_chart():
    """The current lane's dashboard simulation, over the next 60 minutes."""
    try:
        return _shared_forecast_chart(60)
    except Exception as exc:
        raise _http_error(exc)


@router.get("/day-wait-chart")
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
        raise _http_error(exc)


@router.get("/forecast-chart-3h")
def forecast_chart_3h():
    try:
        return _shared_forecast_chart(180)
    except Exception as exc:
        raise _http_error(exc)


@router.get("/forecast-chart-12h")
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
        raise _http_error(exc)


@router.get("/forecast-chart-2d")
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
        raise _http_error(exc)
