"""KPI export: preview (JSON) and CSV download for a day, week or month."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..db import _http_error
from ..errors import ServiceError
from ..services import exports as service

router = APIRouter()

# Only the service's own refusals (400) are turned into answers here. A database failure
# is left to FastAPI (500 without details; 503 "busy" through the app's PoolBusy handler).


@router.get("/export/preview")
def export_preview(period: str, date: str, kpis: Optional[str] = None, lang: str = "fr"):
    try:
        return service.preview(period, date, kpis, lang)
    except ServiceError as exc:
        raise _http_error(exc)


@router.get("/export")
def export_csv(period: str, date: str, kpis: Optional[str] = None, format: str = "csv", lang: str = "fr"):
    try:
        csv_bytes, filename = service.csv_file(period, date, kpis, format, lang)
    except ServiceError as exc:
        raise _http_error(exc)
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
