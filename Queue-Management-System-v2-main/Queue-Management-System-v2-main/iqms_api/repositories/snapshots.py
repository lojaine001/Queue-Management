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


def last_seen(cur, camera_ids) -> dict:
    """{camera_id: newest snapshot time} over the last day (recent chunks only); a camera
    without a snapshot in that day is absent. Works with plain and dict cursors."""
    cur.execute("""
        SELECT camera_id, max(timestamp) AS last_seen FROM queue_state_snapshots
        WHERE camera_id = ANY(%s) AND timestamp > now() - interval '1 day'
        GROUP BY camera_id
    """, (list(camera_ids),))
    return {(r["camera_id"] if isinstance(r, dict) else r[0]): (r["last_seen"] if isinstance(r, dict) else r[1])
            for r in cur.fetchall()}
