"""Operator commands (key-protected): open lanes and answers to alerts."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..db import _http_error
from ..schemas import AlertResponse, SetLanesRequest
from ..security import require_api_key
from ..services import administration as service

router = APIRouter()


@router.post("/set-lanes", dependencies=[Depends(require_api_key)])
def set_lanes(body: SetLanesRequest):
    try:
        return service.set_lanes(body.lanes)
    except Exception as exc:
        raise _http_error(exc)


@router.post("/alert-response", dependencies=[Depends(require_api_key)])
def post_alert_response(body: AlertResponse):
    try:
        return service.record_alert_response(body.response, body.lane_id)
    except Exception as exc:
        raise _http_error(exc)
