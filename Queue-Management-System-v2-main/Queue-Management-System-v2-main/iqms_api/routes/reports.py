"""Day recap and the background traffic / demographic statistics."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter

from ..db import _http_error
from ..services import reports as service

router = APIRouter()


@router.get("/day-recap")
def day_recap(date: Optional[str] = None):
    """The summary for ?date=YYYY-MM-DD, or for the most recent day with data."""
    try:
        return service.day_recap(date)
    except Exception as exc:
        raise _http_error(exc)


@router.get("/statistics/summary")
def statistics_summary():
    """Background demographic/traffic snapshot, with its own freshness timestamp.
    A database failure is left to FastAPI (500 without details), as before."""
    return service.statistics_summary()
