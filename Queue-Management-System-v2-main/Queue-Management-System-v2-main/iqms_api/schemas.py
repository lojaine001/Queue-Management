"""Request bodies of the endpoints that change data."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class AlertResponse(BaseModel):
    response: str
    lane_id: Optional[str] = None


class SetLanesRequest(BaseModel):
    lanes: int


class PushKeys(BaseModel):
    p256dh: str
    auth: str


class PushSubscribeRequest(BaseModel):
    endpoint: str
    keys: PushKeys
    # The device's own alert setting (its threshold slider + selected
    # horizon), sent along with the subscription so a server-side push can
    # fire at exactly the same point this device's in-page alert would --
    # not a separate, possibly-disagreeing threshold.
    threshold_min: float
    horizon_min: int


class PushUnsubscribeRequest(BaseModel):
    endpoint: str
