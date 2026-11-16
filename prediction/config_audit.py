"""Configuration change log (table config_changes, migration 0002): who changed which
setting, when, from what to what.

Used by the API (/set-lanes) and the dashboard (lane selector, forecast settings). Every
function takes the cursor of the transaction that makes the change, so the change and
its record are committed together or not at all. Only real changes are recorded:
setting a value to what it already was adds nothing.
"""
from __future__ import annotations

import psycopg2.extras

SOURCES = ("api", "dashboard")
MAX_ACTOR_LEN = 200


def _json(value):
    return psycopg2.extras.Json(value) if value is not None else None


def record(cur, source: str, actor, setting: str, old, new, settings_revision=None) -> bool:
    """One row when `setting` goes from `old` to `new`; nothing when they are equal."""
    if old == new:
        return False
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    cur.execute(
        "INSERT INTO config_changes (source, actor, setting, old_value, new_value, settings_revision) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (source, (str(actor)[:MAX_ACTOR_LEN] if actor else None), setting, _json(old), _json(new),
         settings_revision),
    )
    return True


def record_settings(cur, source: str, actor, old_config: dict, new_config: dict, settings_revision=None) -> list:
    """One row per forecast setting whose value changed; returns their names."""
    changed = []
    for key in sorted(set(old_config) | set(new_config)):
        if record(cur, source, actor, key, old_config.get(key), new_config.get(key), settings_revision):
            changed.append(key)
    return changed


def recent(cur, limit: int = 50) -> list:
    """The newest changes first."""
    cur.execute(
        "SELECT changed_at, source, actor, setting, old_value, new_value, settings_revision "
        "FROM config_changes ORDER BY changed_at DESC, id DESC LIMIT %s",
        (int(limit),),
    )
    return cur.fetchall()
