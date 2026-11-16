"""Published forecasts: current waits per lane scenario, horizons and charts."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from ..db import _http_error
from ..services import forecasts as service

router = APIRouter()


@router.get("/forecast")
def forecast():
    """The lane simulations published by the background worker."""
    try:
        return service.forecast()
    except Exception as exc:
        raise _http_error(exc)


@router.get("/forecast/wait")
def forecast_wait(minutes: float = 0):
    """Read a horizon relative to the shared forecast's calculation time."""
    try:
        return service.wait_at(minutes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise _http_error(exc)


@router.get("/forecast-chart")
def forecast_chart():
    """The current lane's dashboard simulation, over the next 60 minutes."""
    try:
        return service.chart(60)
    except Exception as exc:
        raise _http_error(exc)


@router.get("/day-wait-chart")
def day_wait_chart(date: Optional[str] = None):
    """A whole day's predicted arrivals and wait (?date=YYYY-MM-DD, default today)."""
    try:
        return service.day_wait_chart(date)
    except Exception as exc:
        raise _http_error(exc)


@router.get("/forecast-chart-3h")
def forecast_chart_3h():
    try:
        return service.chart(180)
    except Exception as exc:
        raise _http_error(exc)


@router.get("/forecast-chart-12h")
def forecast_chart_12h():
    """Last 6 hours of actual entrance counts (3-min slots) for the app chart."""
    try:
        return service.entries_chart("6 hours", "%H:%M")
    except Exception as exc:
        raise _http_error(exc)


@router.get("/forecast-chart-2d")
def forecast_chart_2d():
    """Last 2 days of actual entrance counts (3-min slots) for the app chart."""
    try:
        return service.entries_chart("2 days", "%d/%m %H:%M")
    except Exception as exc:
        raise _http_error(exc)
