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
