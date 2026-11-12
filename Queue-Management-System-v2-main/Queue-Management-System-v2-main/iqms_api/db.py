"""The shared database connection pool and how endpoints report failures."""
from __future__ import annotations

import os

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .db_pool import DbPool, PoolBusy
from .settings import DB_CONFIG, STORE_TZ

# One shared pool instead of a new connection per request (see db_pool.py).
# Each connection is opened once with the store time zone and a query time limit.
DB_POOL = DbPool(
    {**DB_CONFIG, "connect_timeout": 5, "application_name": "iqms-api"},  # visible in pg_stat_activity
    minconn=int(os.getenv("DB_POOL_MIN", 5)),   # kept open; psycopg2 closes extras above this when returned
    maxconn=int(os.getenv("DB_POOL_MAX", 10)),
    wait_sec=float(os.getenv("DB_POOL_WAIT_SEC", 5)),
    session_options=f"-c timezone={STORE_TZ} -c statement_timeout={int(os.getenv('DB_STATEMENT_TIMEOUT_MS', 15000))}",
)


def _conn():
    """A pooled connection: `with _conn() as conn:` commits on success, rolls back on
    error, and always gives the connection back."""
    return DB_POOL.connection()


def _http_error(exc: Exception) -> HTTPException:
    """What an endpoint answers when something fails: 503 while the pool is busy, else 500."""
    if isinstance(exc, PoolBusy):
        return HTTPException(status_code=503, detail="The database is busy; try again in a few seconds.",
                             headers={"Retry-After": "5"})
    return HTTPException(status_code=500, detail=str(exc))


def _pool_busy_handler(request: Request, exc: PoolBusy):
    err = _http_error(exc)
    return JSONResponse(status_code=err.status_code, content={"detail": err.detail}, headers=err.headers)
