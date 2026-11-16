"""queue_predictions: the ensemble's arrivals and wait per slot, one row per run and slot."""
from __future__ import annotations


def for_day(cur, day) -> list[dict]:
    """Every slot of `day`, from the newest run that predicted it (missing values = 0)."""
    cur.execute("""
        SELECT DISTINCT ON (prediction_for)
            prediction_for,
            COALESCE(ensemble_yhat, 0)    AS arrivals,
            COALESCE(est_wait_minutes, 0) AS wait_min
        FROM queue_predictions
        WHERE DATE(prediction_for) = %s
        ORDER BY prediction_for ASC, predicted_at DESC
    """, (day,))
    return cur.fetchall()


def alert_slots_on(cur, day) -> int:
    """Slots predicted in ALERT on `day` (None: today), any run."""
    if day is None:
        condition, params = "DATE(prediction_for) = CURRENT_DATE", ()
    else:
        condition, params = "DATE(prediction_for) = %s", (day,)
    cur.execute(f"""
        SELECT COUNT(*) AS alert_slots
        FROM queue_predictions
        WHERE {condition}
          AND status = 'ALERT'
    """, params)
    row = cur.fetchone()
    return int((row["alert_slots"] or 0) if row else 0)
