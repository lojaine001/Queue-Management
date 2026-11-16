"""Web push subscriptions: a device's own alert setting, stored so the push checker
can alert it at the same point its in-page alert would."""
from __future__ import annotations

from ..db import _conn
from ..repositories import push


def check_table() -> None:
    """Raises when push_subscriptions is missing (database not migrated)."""
    with _conn() as conn:
        with conn.cursor() as cur:
            push.require_table(cur)


def subscribe(endpoint: str, p256dh: str, auth: str, threshold_min: float, horizon_min: int) -> dict:
    with _conn() as conn:
        with conn.cursor() as cur:
            push.upsert(cur, endpoint, p256dh, auth, threshold_min, horizon_min)
        conn.commit()
    return {"status": "subscribed"}


def unsubscribe(endpoint: str) -> dict:
    with _conn() as conn:
        with conn.cursor() as cur:
            push.delete(cur, endpoint)
        conn.commit()
    return {"status": "unsubscribed"}
