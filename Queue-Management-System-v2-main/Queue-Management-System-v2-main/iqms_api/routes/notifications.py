"""Web push: the public VAPID key and (key-protected) subscribe / unsubscribe."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from ..db import _http_error
from ..schemas import PushSubscribeRequest, PushUnsubscribeRequest
from ..security import require_api_key
from ..services import notifications as service

router = APIRouter()


try:
    service.check_table()
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
        return service.subscribe(body.endpoint, body.keys.p256dh, body.keys.auth,
                                 body.threshold_min, body.horizon_min)
    except Exception as exc:
        raise _http_error(exc)


@router.post("/push-unsubscribe", dependencies=[Depends(require_api_key)])
def push_unsubscribe(body: PushUnsubscribeRequest):
    try:
        return service.unsubscribe(body.endpoint)
    except Exception as exc:
        raise _http_error(exc)
