"""Web push: the public VAPID key and (key-protected) subscribe / unsubscribe."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from ..db import _conn, _http_error
from ..schemas import PushSubscribeRequest, PushUnsubscribeRequest
from ..security import require_api_key
from prediction.schema_check import require_tables

router = APIRouter()


def _check_push_table():
    with _conn() as conn:
        with conn.cursor() as cur:
            require_tables(cur, "push_subscriptions")


try:
    _check_push_table()
except Exception as exc:
    # Don't crash the whole API on startup (database not reachable yet, or not
    # migrated): the push endpoints fail per request instead, like every other one.
    print(f"[push] push_subscriptions not usable: {exc}")


@router.get("/vapid-public-key")
def vapid_public_key():
    key = os.getenv("VAPID_PUBLIC_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="VAPID_PUBLIC_KEY not configured on the server")
    return {"public_key": key}


@router.post("/push-subscribe", dependencies=[Depends(require_api_key)])
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
        raise _http_error(exc)


@router.post("/push-unsubscribe", dependencies=[Depends(require_api_key)])
def push_unsubscribe(body: PushUnsubscribeRequest):
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM push_subscriptions WHERE endpoint = %s", (body.endpoint,))
            conn.commit()
        return {"status": "unsubscribed"}
    except Exception as exc:
        raise _http_error(exc)
