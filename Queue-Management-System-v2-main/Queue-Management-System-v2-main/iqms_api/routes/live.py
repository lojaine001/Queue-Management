"""Live checkout lanes and the current alerts."""
from __future__ import annotations

from fastapi import APIRouter

from ..db import _http_error
from ..services import live as service

router = APIRouter()


@router.get("/live-lanes")
def live_lanes():
    """Per-lane queue status for the Live tab."""
    try:
        return service.live_lanes()
    except Exception as exc:
        raise _http_error(exc)


@router.get("/alerts")
def get_alerts():
    """The current alert level, from the published forecast."""
    try:
        return service.current_alert()
    except Exception as exc:
        raise _http_error(exc)
