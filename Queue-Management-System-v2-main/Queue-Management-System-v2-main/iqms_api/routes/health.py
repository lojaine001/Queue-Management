"""GET /health: liveness plus the database pool's state."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from ..db import DB_POOL

router = APIRouter()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    # db_pool: connections in use out of max, requests waiting, requests refused as busy,
    # broken connections replaced. in_use == max for long = something holds connections.
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat(), "db_pool": DB_POOL.stats()}
