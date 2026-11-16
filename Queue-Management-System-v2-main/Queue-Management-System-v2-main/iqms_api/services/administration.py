"""Operator commands: open lanes and answers to alerts."""
from __future__ import annotations

import psycopg2
import psycopg2.extras

from prediction import config_audit
from prediction.forecast_state import finite, freshness, scenario_for

from ..db import _conn
from ..errors import BadRequest, Conflict
from ..repositories import alerts, dashboard


def set_lanes(lanes: int, actor=None) -> dict:
    """Switch the published waits to the forecast's scenario for `lanes` open lanes.

    Refused while the background forecast is stale or has no scenario for that count:
    the waits would describe an old situation. A real change is recorded in
    config_changes (who = `actor`), in the same transaction."""
    if not (1 <= lanes <= 4):
        raise BadRequest("lanes must be between 1 and 4")
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            state = dashboard.lock_state(cur)
            scenario = scenario_for(state, lanes)
            if not scenario or freshness(state)["stale"]:
                raise Conflict("Wait for a fresh background forecast before changing lanes.")
            old_lanes = state.get("open_lanes")
            dashboard.set_open_lanes(
                cur, lanes, finite(scenario.get("wait_0m")), finite(scenario.get("wait_5m")),
                finite(scenario.get("wait_10m")), finite(scenario.get("wait_15m")),
            )
            config_audit.record(cur, "api", actor, "open_lanes", old_lanes, lanes)
        conn.commit()
    # Changing the lane selection does not make the simulation's inputs newer.
    return {"status": "ok", "lanes": lanes, **freshness(state)}


ALERT_RESPONSES = ("opening_lane", "cannot_open", "false_alarm")


def record_alert_response(response: str, lane_id) -> dict:
    if response not in ALERT_RESPONSES:
        # Same text as before, but in a fixed order (it used to print a Python set,
        # whose order changed every time the API started).
        allowed = ", ".join(repr(r) for r in ALERT_RESPONSES)
        raise BadRequest(f"response must be one of {{{allowed}}}")
    with _conn() as conn:
        with conn.cursor() as cur:
            alerts.insert_response(cur, response, lane_id)
        conn.commit()
    return {"status": "recorded", "response": response}
