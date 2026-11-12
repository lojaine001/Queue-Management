"""Settings read once at startup from the project's .env, plus the import paths.

Required secrets have no default: the API stops at startup with a clear message
if one is missing (IQMS_API_KEY first, then DB_PASSWORD).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Every other script in this codebase (dashboard.py, ensemble_predict.py,
# run_scheduler.py, push_alert_checker.py, ...) loads .env this way --
# this file never did, so anything set only in .env (not a real OS-level
# env var) was silently invisible to it. Caught via VAPID_PUBLIC_KEY
# returning "not configured" despite being set in .env.
load_dotenv(find_dotenv(usecwd=True))

# API_DIR = the folder holding api.py and this package; ROOT_DIR = the project root
# (for `prediction`). Sibling modules (camera_snapshot) must import even when api.py
# is loaded by file path (the unit tests do this).
API_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = API_DIR.parents[1]
for _path in (ROOT_DIR, API_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SNAP_DIR = API_DIR / "snapshots"

# Endpoints that change data require this key in the X-API-Key header.
# Read-only (GET) endpoints stay open. The web app never holds the key: the
# Caddy proxy (IQMS-Web/Caddyfile) adds it to the two push calls it forwards.
API_KEY = os.getenv("IQMS_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError(
        "IQMS_API_KEY is not set. Add a long random value to .env (see .env.example), "
        "e.g. python -c \"import secrets; print(secrets.token_urlsafe(32))\""
    )


def cors_allowed_origins() -> list[str]:
    """CORS: which OTHER websites may call this API from a browser. Empty (the
    default) = none. The web app does not need an entry: Caddy serves it and
    /api from the same address. Native apps (IQMSManager) and scripts are not
    browsers, so CORS does not apply to them. Never "*"."""
    origins = [o.strip().rstrip("/") for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if "*" in origins:
        raise RuntimeError("CORS_ALLOWED_ORIGINS must list explicit origins (https://host), not '*'.")
    return origins


def _required_env(name: str) -> str:
    """A secret with no default: stop at startup with a clear message if it is missing."""
    value = os.getenv(name, "").strip()
    if not value or value == "change-me":
        raise RuntimeError(f"{name} is not set. Add it to .env (see .env.example).")
    return value


DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "iqms"),
    user=os.getenv("DB_USER", "postgres"),
    password=_required_env("DB_PASSWORD"),
)

LANE_MAX_CAPACITY = 10  # denominator for fill bar
STORE_TZ = os.getenv("STORE_TZ", "Europe/Paris")
CHECKOUT_CAM_ID = os.getenv("CHECKOUT_CAM_ID", "Bosch_Camera_exit")
BUCKET_MIN = int(os.getenv("BUCKET_MINUTES", 3))
