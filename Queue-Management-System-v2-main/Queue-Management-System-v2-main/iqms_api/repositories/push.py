"""push_subscriptions: one row per device subscribed to wait alerts."""
from __future__ import annotations

from prediction.schema_check import require_tables


def require_table(cur) -> None:
    require_tables(cur, "push_subscriptions")


def upsert(cur, endpoint: str, p256dh: str, auth: str, threshold_min: float, horizon_min: int) -> None:
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
    """, (endpoint, p256dh, auth, threshold_min, horizon_min))


def delete(cur, endpoint: str) -> None:
    cur.execute("DELETE FROM push_subscriptions WHERE endpoint = %s", (endpoint,))
