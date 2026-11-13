"""Health endpoints.

- GET /health          the API process is up (+ the database pool's state). Docker checks this.
- GET /health/ready    the API can serve requests: database reachable and at the migration
                       version this code expects. 200 ready / 503 not ready.
- GET /health/workers  every background part is still producing (cameras, scheduler,
                       forecast, push alerts). 200 all ok / 503 otherwise.

These answer through the public link too (every GET does), so they report statuses and
times only, never error messages; the details stay in the logs.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from prediction.runtime import ROOT, RUNTIME, read_json

from .. import health_checks as hc
from ..db import DB_POOL, _conn
from ..settings import CHECKOUT_CAM_ID, ENTRANCE_CAM_ID

router = APIRouter()

MIGRATIONS_DIR = ROOT / "db" / "migrations" / "versions"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    # db_pool: connections in use out of max, requests waiting, requests refused as busy,
    # broken connections replaced. in_use == max for long = something holds connections.
    return {"status": "ok", "time": _now().isoformat(), "db_pool": DB_POOL.stats()}


@router.get("/health/ready")
def health_ready():
    expected = hc.expected_schema_version(MIGRATIONS_DIR)
    body = {"status": "ready", "time": _now().isoformat(), "schema_expected": expected, "schema_version": None}
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.alembic_version') IS NOT NULL")
                if cur.fetchone()[0]:
                    cur.execute("SELECT version_num FROM alembic_version")
                    row = cur.fetchone()
                    body["schema_version"] = row[0] if row else None
    except Exception as exc:
        body.update(status="not_ready", reason=f"database not reachable ({type(exc).__name__})")
        return JSONResponse(status_code=503, content=body)

    if body["schema_version"] is None:
        body.update(status="not_ready", reason="database has no migration version; see db/README.md")
    elif expected and body["schema_version"] != expected:
        body.update(status="not_ready",
                    reason=f"database at {body['schema_version']}, code expects {expected}: "
                           "run alembic -c db/alembic.ini upgrade head")
    return JSONResponse(status_code=200 if body["status"] == "ready" else 503, content=body)


@router.get("/health/workers")
def health_workers():
    now = _now()
    scheduler_state = read_json(RUNTIME / "scheduler.json")
    push_state = read_json(RUNTIME / "push_alerts.json")
    workers = {
        "scheduler": hc.scheduler_check(scheduler_state, now),
        "push_alerts": hc.check(".runtime/push_alerts.json heartbeat", push_state.get("heartbeat"),
                                hc.PUSH_MAX_AGE_SEC, now),
    }
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                # Only the last day is searched (recent chunks); older = "missing".
                cur.execute("""
                    SELECT camera_id, max(timestamp) FROM queue_state_snapshots
                    WHERE camera_id = ANY(%s) AND timestamp > now() - interval '1 day'
                    GROUP BY camera_id
                """, ([CHECKOUT_CAM_ID, ENTRANCE_CAM_ID],))
                last_snapshot = dict(cur.fetchall())
                cur.execute("SELECT max(published_at) FROM wait_forecast_runs "
                            "WHERE published_at > now() - interval '1 day'")
                last_forecast = cur.fetchone()[0]
    except Exception as exc:
        body = {"status": "degraded", "time": now.isoformat(), "workers": workers,
                "reason": f"database not reachable ({type(exc).__name__})"}
        return JSONResponse(status_code=503, content=body)

    for name, camera_id in (("camera_checkout", CHECKOUT_CAM_ID), ("camera_entrance", ENTRANCE_CAM_ID)):
        workers[name] = hc.check("queue_state_snapshots", last_snapshot.get(camera_id), hc.CAMERA_MAX_AGE_SEC,
                                 now, camera_id=camera_id)
    workers["forecast"] = hc.check("wait_forecast_runs", last_forecast,
                                   hc.forecast_max_age_sec(scheduler_state.get("interval_min")), now)

    workers = dict(sorted(workers.items()))
    all_ok = all(w["status"] == hc.OK for w in workers.values())
    body = {"status": "ok" if all_ok else "degraded", "time": now.isoformat(), "workers": workers}
    return JSONResponse(status_code=200 if all_ok else 503, content=body)
