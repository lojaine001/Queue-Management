"""Why a service refused a request. Services raise these instead of HTTP errors;
db._http_error turns them into the answer (status code + detail)."""
from __future__ import annotations


class ServiceError(Exception):
    status_code = 500

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class BadRequest(ServiceError):
    """The request itself is wrong (out-of-range value, unknown option): 400."""
    status_code = 400


class Conflict(ServiceError):
    """The request is valid but cannot be done in the current state: 409."""
    status_code = 409
