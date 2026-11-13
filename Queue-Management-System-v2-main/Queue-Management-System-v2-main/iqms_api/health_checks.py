"""Signs of life for /health/ready and /health/workers (no web or database code here).

Each background part leaves a regular trace; a part is "ok" while its last trace is
recent enough, "stale" when it is older, "missing" when there is none:

- cameras: a row in queue_state_snapshots every 10 s (snapshot_interval), day and night
- scheduler: .runtime/scheduler.json "heartbeat", rewritten every second while it runs
- forecast: the newest wait_forecast_runs row (a scheduler that is alive but whose runs
  keep failing publishes nothing)
- push alerts: .runtime/push_alerts.json "heartbeat", rewritten at every check (30 s)
"""
from __future__ import annotations

import ast
import os
from datetime import datetime, timezone
from pathlib import Path

OK, STALE, MISSING = "ok", "stale", "missing"


def _env_seconds(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


CAMERA_MAX_AGE_SEC = _env_seconds("HEALTH_CAMERA_MAX_AGE_SEC", 60)        # 6 missed snapshots
SCHEDULER_MAX_AGE_SEC = _env_seconds("HEALTH_SCHEDULER_MAX_AGE_SEC", 120)  # same as Docker's check
PUSH_MAX_AGE_SEC = _env_seconds("HEALTH_PUSH_MAX_AGE_SEC", 120)            # 4 missed checks


def forecast_max_age_sec(interval_min: float | None) -> float:
    """Three missed runs, and never under 10 minutes (one slow run is not an outage)."""
    override = os.getenv("HEALTH_FORECAST_MAX_AGE_SEC")
    if override:
        return _env_seconds("HEALTH_FORECAST_MAX_AGE_SEC", 600)
    interval = interval_min or float(os.getenv("PREDICTION_INTERVAL_MIN", 15))
    return max(600.0, 3 * 60 * float(interval))


def parse_time(value) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def age_seconds(when, now: datetime) -> float | None:
    when = parse_time(when)
    if when is None:
        return None
    if when.tzinfo is None:          # stored without a zone = UTC (heartbeat files are UTC)
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (now - when).total_seconds())


def classify(age: float | None, max_age: float) -> str:
    if age is None:
        return MISSING
    return OK if age <= max_age else STALE


def check(name_of_trace: str, when, max_age: float, now: datetime, **extra) -> dict:
    age = age_seconds(when, now)
    out = {"status": classify(age, max_age), "trace": name_of_trace,
           "last_seen": parse_time(when).isoformat() if parse_time(when) else None,
           "age_sec": round(age, 1) if age is not None else None, "max_age_sec": max_age}
    out.update(extra)
    return out


def scheduler_check(state: dict, now: datetime) -> dict:
    """The scheduler process: heartbeat recent and not stopped on purpose.

    Its "status" also describes the last forecast run ("failed" = that run failed; the
    scheduler keeps going and tries again next interval), so a failed run does not make
    the scheduler unhealthy: it shows up under "forecast" (nothing new published)."""
    result = check(".runtime/scheduler.json heartbeat", state.get("heartbeat"), SCHEDULER_MAX_AGE_SEC, now,
                   last_run=state.get("status"), last_success=state.get("last_success"),
                   last_exit_code=state.get("exit_code"))
    if result["status"] == OK and state.get("status") == "stopped":
        result["status"] = STALE
    return result


def expected_schema_version(versions_dir: Path) -> str | None:
    """The newest migration in db/migrations/versions: the one no other migration
    revises. None when the folder is missing or the chain is not a single line."""
    revisions, parents = set(), set()
    try:
        files = sorted(Path(versions_dir).glob("*.py"))
    except OSError:
        return None
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("revision", "down_revision"):
                    try:
                        values[node.targets[0].id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
        if values.get("revision"):
            revisions.add(values["revision"])
            down = values.get("down_revision")
            parents.update(down if isinstance(down, (tuple, list)) else [down] if down else [])
    heads = revisions - parents
    return heads.pop() if len(heads) == 1 else None
