"""dashboard_state: the single row (id = 1) the forecast publishes and the lane choice updates."""
from __future__ import annotations


def forecast_state(cur) -> dict:
    """The row plus the revision of the settings now saved (to tell a forecast made with
    older settings)."""
    cur.execute("SELECT d.*, f.revision AS current_settings_revision FROM dashboard_state d LEFT JOIN forecast_settings f ON f.id=d.id WHERE d.id=1")
    return cur.fetchone() or {}


def live_inputs(cur):
    """People waiting now, service time and the 5-minute wait (or None before the first forecast)."""
    cur.execute("SELECT queue_now, service_min, wait_5m FROM dashboard_state WHERE id = 1")
    return cur.fetchone()


def wait_in_15_min(cur):
    """The forecast wait in 15 minutes and when it was published (or None)."""
    cur.execute("SELECT wait_15m, updated_at FROM dashboard_state WHERE id = 1")
    return cur.fetchone()


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
