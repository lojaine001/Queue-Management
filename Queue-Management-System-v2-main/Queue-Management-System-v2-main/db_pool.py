"""
Bounded PostgreSQL connection pool for the API.

Before: every request opened a new connection, and `with psycopg2.connect() as
conn:` only commits or rolls back, it does not close, so connections stayed
open until garbage collection. Now connections are opened once and reused:

    with pool.connection() as conn:     # waits up to wait_sec for a free one
        ...                             # commit on success, rollback on error,
                                        # always returned (broken ones replaced)

When all connections are busy for longer than wait_sec, PoolBusy is raised
(the API turns it into 503). stats() is shown by /health.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool as pg_pool

log = logging.getLogger("iqms.api.db")


class PoolBusy(Exception):
    """Every connection stayed in use for longer than the wait limit."""


class DbPool:
    def __init__(self, conn_kwargs: dict, minconn: int = 1, maxconn: int = 10, wait_sec: float = 5.0,
                 session_options: str = ""):
        if minconn < 0 or maxconn < 1 or minconn > maxconn:
            raise ValueError(f"invalid pool size min={minconn} max={maxconn}")
        self.minconn, self.maxconn, self.wait_sec = minconn, maxconn, wait_sec
        # Session settings are sent once when a connection opens, not per request.
        self._kwargs = {**conn_kwargs, "options": " ".join(filter(None, [conn_kwargs.get("options", ""), session_options]))}
        self._pool: pg_pool.ThreadedConnectionPool | None = None
        self._create_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(maxconn)   # never more than maxconn handed out
        self._count_lock = threading.Lock()
        self._in_use = 0
        self._waiting = 0
        self._busy_rejections = 0
        self._replaced = 0

    def _get_pool(self) -> pg_pool.ThreadedConnectionPool:
        # Created on first use, so importing the API does not need the database.
        if self._pool is None:
            with self._create_lock:
                if self._pool is None:
                    self._pool = pg_pool.ThreadedConnectionPool(self.minconn, self.maxconn, **self._kwargs)
        return self._pool

    def _count(self, name: str, delta: int) -> None:
        with self._count_lock:
            setattr(self, name, getattr(self, name) + delta)

    @contextmanager
    def connection(self):
        self._count("_waiting", 1)
        started = time.monotonic()
        try:
            got_slot = self._slots.acquire(timeout=self.wait_sec)
        finally:
            self._count("_waiting", -1)
        if not got_slot:
            self._count("_busy_rejections", 1)
            log.warning("DB pool busy: all %d connections in use for %.1fs", self.maxconn, time.monotonic() - started)
            raise PoolBusy(f"all {self.maxconn} database connections are busy")
        conn = None
        broken = False
        try:
            conn = self._get_pool().getconn()
            self._count("_in_use", 1)
            yield conn
            conn.commit()
        except BaseException:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    broken = True   # e.g. the server dropped the connection
            raise
        finally:
            if conn is not None:
                broken = broken or bool(conn.closed)
                if broken:
                    self._count("_replaced", 1)
                try:
                    self._pool.putconn(conn, close=broken)
                except Exception:
                    log.exception("Could not return a connection to the pool")
                self._count("_in_use", -1)
            self._slots.release()

    def stats(self) -> dict:
        with self._count_lock:
            return {"in_use": self._in_use, "max": self.maxconn, "waiting": self._waiting,
                    "busy_rejections": self._busy_rejections, "replaced": self._replaced,
                    "open": self._pool is not None}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
