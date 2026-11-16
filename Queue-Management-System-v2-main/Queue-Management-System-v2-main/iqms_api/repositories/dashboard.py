"""dashboard_state: the single row (id = 1) the forecast publishes and the lane choice updates."""
from __future__ import annotations


def lock_state(cur) -> dict:
    """The row, locked until the transaction ends (so two lane changes cannot interleave)."""
    cur.execute("SELECT * FROM dashboard_state WHERE id = 1 FOR UPDATE")
    return cur.fetchone() or {}


def set_open_lanes(cur, lanes: int, wait_0m, wait_5m, wait_10m, wait_15m) -> None:
    cur.execute(
        """UPDATE dashboard_state
           SET open_lanes = %s, wait_0m = %s, wait_5m = %s, wait_10m = %s, wait_15m = %s
           WHERE id = 1""",
        (lanes, wait_0m, wait_5m, wait_10m, wait_15m),
    )
