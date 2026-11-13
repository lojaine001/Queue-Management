"""Operator commands (key-protected): open lanes and answers to alerts."""
from __future__ import annotations

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException

from prediction.forecast_state import finite, freshness, scenario_for

from ..db import _conn, _http_error
from ..schemas import AlertResponse, SetLanesRequest
from ..security import require_api_key

router = APIRouter()


@router.post("/set-lanes", dependencies=[Depends(require_api_key)])
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
        raise _http_error(exc)


@router.post("/alert-response", dependencies=[Depends(require_api_key)])
def post_alert_response(body: AlertResponse):
    valid = {"opening_lane", "cannot_open", "false_alarm"}
    if body.response not in valid:
        raise HTTPException(status_code=400, detail=f"response must be one of {valid}")
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO alert_responses (response, lane_id) VALUES (%s, %s)",
                    (body.response, body.lane_id),
                )
            conn.commit()
        return {"status": "recorded", "response": body.response}
    except Exception as exc:
        raise _http_error(exc)
