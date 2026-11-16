"""alert_responses: what staff answered to a queue alert."""
from __future__ import annotations


def insert_response(cur, response: str, lane_id) -> None:
    cur.execute(
        "INSERT INTO alert_responses (response, lane_id) VALUES (%s, %s)",
        (response, lane_id),
    )
