"""Published forecasts: current waits per lane scenario, any horizon, and the charts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from prediction.forecast_state import finite, freshness, lane_scenarios, scenario_for, wait_at_horizon

from ..db import _conn
from ..errors import BadRequest
from ..repositories import dashboard, entrance, predictions
from ..settings import STORE_TZ


def current_state() -> dict:
    """The forecast the background worker published last (dashboard_state)."""
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            return dashboard.forecast_state(cur)


def forecast() -> dict:
    """The waits now and in 5/10/15 minutes for the open lanes, and every lane scenario."""
    state = current_state()
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


def wait_at(minutes: float) -> dict:
    """The wait `minutes` after the forecast's calculation time.
    Raises ValueError for a horizon the forecast does not cover (the route answers 422)."""
    return wait_at_horizon(current_state(), minutes)


def chart(horizon_minutes) -> dict:
    """The open lanes' simulated slots from now until `horizon_minutes` ahead."""
    state = current_state()
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


def day_wait_chart(date) -> dict:
    """
    Full-day history of predicted arrivals/wait for the Statistique page's
    Temps d'attente chart. Unlike the forecast chart (always the next 60 min
    forward from now), this covers a whole day - past or current - at
    native prediction resolution. For today, it naturally tapers off
    wherever the ensemble job's predictions currently stop (it doesn't
    force the line to stop exactly at "now" or fabricate future points).
    """
    if date:
        try:
            ref_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise BadRequest("date must be in YYYY-MM-DD format")
    else:
        ref_date = datetime.now().date()

    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            rows = predictions.for_day(cur, ref_date)

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


def entries_chart(since: str, time_format: str) -> dict:
    """Actual entrance counts per 3-minute slot over the last `since` (e.g. '6 hours')."""
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            rows = entrance.entries_per_3_minutes(cur, since)
    return {
        "slots": [
            {
                "time":    row["slot"].strftime(time_format),
                "entries": int(row["entries"]),
            }
            for row in rows
        ]
    }
