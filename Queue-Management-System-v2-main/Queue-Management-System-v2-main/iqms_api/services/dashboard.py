"""Everything the Live screen shows, in one answer read at one moment (API v1).

Built from the same services as the separate endpoints (/live-lanes, /alerts, /forecast)
and the same camera check as /health/workers, so the values match them; the
difference is that all of it is read in one read-only transaction (one snapshot of
the database), and the alert level is given for every horizon so apps don't
recompute it with their own thresholds."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from .. import health_checks
from ..db import _conn
from ..repositories import dashboard as dashboard_repo
from ..repositories import snapshots
from ..settings import CHECKOUT_CAM_ID, ENTRANCE_CAM_ID, STORE_TZ
from . import forecasts, live

API_VERSION = 1
HORIZONS = (("wait_now_min", 0), ("wait_5_min", 5), ("wait_10_min", 10), ("wait_15_min", 15))


def _camera(last_seen, now) -> dict:
    """ok / stale / missing and the age of the camera's newest snapshot (no camera id, no error text)."""
    check = health_checks.check("queue_state_snapshots", last_seen, health_checks.CAMERA_MAX_AGE_SEC, now)
    return {"status": check["status"], "age_sec": check["age_sec"]}


def live_dashboard() -> dict:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # One snapshot of the database for every read below; this transaction only
            # (the pool ends it when the request is done).
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            lanes = live.read_live_lanes(cur)
            alert = live.read_alert(cur)
            state = dashboard_repo.forecast_state(cur)
            seen = snapshots.last_seen(cur, [CHECKOUT_CAM_ID, ENTRANCE_CAM_ID])
    now = datetime.now(timezone.utc)

    forecast = forecasts.forecast_from_state(state)
    waits = []
    for key, horizon in HORIZONS:
        wait = forecast.pop(key)
        waits.append({"horizon_min": horizon, "wait_min": wait,
                      "level": live.alert_level(wait)[0] if wait is not None else None})

    return {
        "api_version": API_VERSION,
        "generated_at": now.astimezone(ZoneInfo(STORE_TZ)).isoformat(),
        "lanes": lanes["lanes"],
        "queue": lanes["snapshot"],
        "forecast": {"waits": waits, **forecast},
        "alert": alert,
        "cameras": {"checkout": _camera(seen.get(CHECKOUT_CAM_ID), now),
                    "entrance": _camera(seen.get(ENTRANCE_CAM_ID), now)},
    }
