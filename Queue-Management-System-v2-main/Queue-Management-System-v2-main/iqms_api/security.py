"""The API key check used by every endpoint that changes data."""
from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request

from .settings import API_KEY

auth_log = logging.getLogger("iqms.api.auth")


def require_api_key(request: Request, x_api_key: Optional[str] = Header(default=None)):
    """FastAPI dependency: 401 unless the X-API-Key header matches IQMS_API_KEY."""
    if x_api_key and secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        return
    auth_log.warning("Rejected %s %s from %s: %s API key", request.method, request.url.path,
                     request.client.host if request.client else "?", "wrong" if x_api_key else "no")
    raise HTTPException(status_code=401, detail="Missing or invalid API key",
                        headers={"WWW-Authenticate": "X-API-Key"})
