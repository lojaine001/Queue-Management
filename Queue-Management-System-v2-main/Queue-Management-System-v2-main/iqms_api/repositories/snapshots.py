"""queue_state_snapshots: what each camera counts, every 10 seconds."""
from __future__ import annotations


def latest(cur, camera_id: str):
    """The camera's newest snapshot: people waiting and the count per lane (or None)."""
    cur.execute("""
        SELECT queue_count, lane_counts
        FROM queue_state_snapshots
        WHERE camera_id = %s
        ORDER BY timestamp DESC LIMIT 1
    """, (camera_id,))
    return cur.fetchone()
