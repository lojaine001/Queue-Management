"""API version 1: new endpoints live under /v1 (public address /api/v1/..., Caddy removes /api).

The unversioned endpoints keep working unchanged for the current apps."""
from __future__ import annotations

from fastapi import APIRouter

from ..db import _http_error
from ..services import dashboard as dashboard_service

router = APIRouter(prefix="/v1")


@router.get("/dashboard/live")
def dashboard_live():
    """Everything the Live screen shows, read at one moment: lanes, queue, forecast waits
    with their alert level, the current alert and the cameras' freshness."""
    try:
        return dashboard_service.live_dashboard()
    except Exception as exc:
        raise _http_error(exc)
