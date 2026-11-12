"""Database reads shared by several endpoints."""
from __future__ import annotations

import psycopg2
import psycopg2.extras

from .db import _conn


def _forecast_state():
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT d.*, f.revision AS current_settings_revision FROM dashboard_state d LEFT JOIN forecast_settings f ON f.id=d.id WHERE d.id=1")
            return cur.fetchone() or {}
