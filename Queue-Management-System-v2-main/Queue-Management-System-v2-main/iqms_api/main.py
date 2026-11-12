"""
IQMS API: builds the FastAPI app from the route modules.

Run (start_all.ps1 does this):  uvicorn api:app --host 0.0.0.0 --port 8000

Layout:
    settings.py      .env, required secrets, constants, import paths
    security.py      API key check for the endpoints that change data
    db.py            shared connection pool (db_pool.py) and error mapping
    schemas.py       request bodies
    kpis.py          KPI queries shared by the recap and the export
    repositories.py  database reads shared by several endpoints
    routes/          one module per area; each defines `router`
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import settings
from .db import DB_POOL, PoolBusy, _pool_busy_handler
from .routes import administration, exports, forecasts, health, live, notifications, reports, snapshots

ROUTERS = [health, notifications, live, forecasts, reports, snapshots, administration, exports]


def create_app() -> FastAPI:
    app = FastAPI(title="IQMS Manager API", version="2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
        allow_credentials=False,
    )
    app.add_exception_handler(PoolBusy, _pool_busy_handler)
    app.on_event("shutdown")(DB_POOL.close)  # same hook the single-file api.py used
    for module in ROUTERS:
        app.include_router(module.router)
    return app


app = create_app()
