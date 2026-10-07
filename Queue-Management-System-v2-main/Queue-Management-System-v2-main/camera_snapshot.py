"""
camera_snapshot.py -- fetches a clean (non-overlaid) JPEG snapshot directly
from the checkout camera over HTTPS (HTTP Digest auth), resizes it, and
caches it server-side so every client shares one fetch per refresh window
regardless of how many phones/browsers have the Live tab open.

Camera endpoint discovered empirically (6 Oct 2026): this Bosch camera
serves https://<host>/snap.jpg directly over HTTPS (port 80/HTTP is closed)
-- no ONVIF GetSnapshotUri needed. Credentials come from env vars, never
from the committed Head-Detector/config.yml (that file's RTSP credentials
are already flagged elsewhere as a separate, pre-existing issue).
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
import time

import requests
import urllib3
from requests.auth import HTTPDigestAuth
from PIL import Image

# The camera's cert is self-signed (local device) -- verify=False is
# intentional and scoped to this one internal host, not disabling TLS
# verification globally. Silence the resulting per-request warning.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CHECKOUT_CAM_HOST = os.getenv("CHECKOUT_CAM_HOST", "192.168.1.48")
CHECKOUT_CAM_USER = os.getenv("CHECKOUT_CAM_USER", "service")
CHECKOUT_CAM_PASSWORD = os.getenv("CHECKOUT_CAM_PASSWORD")
SNAPSHOT_HEIGHT = int(os.getenv("SNAPSHOT_HEIGHT", 480))
SNAPSHOT_JPEG_QUALITY = int(os.getenv("SNAPSHOT_JPEG_QUALITY", 70))
SNAPSHOT_REFRESH_SEC = float(os.getenv("SNAPSHOT_REFRESH_SEC", 10))
FETCH_TIMEOUT_SEC = 3

_lock = threading.Lock()
_cache = {"bytes": None, "fetched_at": 0.0, "etag": None}


def _fetch_from_camera() -> bytes:
    url = f"https://{CHECKOUT_CAM_HOST}/snap.jpg"
    resp = requests.get(
        url,
        auth=HTTPDigestAuth(CHECKOUT_CAM_USER, CHECKOUT_CAM_PASSWORD),
        verify=False,
        timeout=FETCH_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.content


def _resize_and_encode(raw: bytes) -> bytes:
    with Image.open(io.BytesIO(raw)) as im:
        # Measured 6 Oct 2026: the camera's own JPEG encoder is already more
        # efficient than re-encoding through Pillow at quality=70 (68796
        # bytes re-encoded vs 32768 bytes native, same 640x480 frame) -- so
        # skip the re-encode entirely whenever the camera already matches
        # the target height. Only resize/re-encode if that ever changes.
        if im.height == SNAPSHOT_HEIGHT:
            return raw
        im = im.convert("RGB")
        ratio = SNAPSHOT_HEIGHT / im.height
        new_width = max(1, round(im.width * ratio))
        im = im.resize((new_width, SNAPSHOT_HEIGHT), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=SNAPSHOT_JPEG_QUALITY)
        return out.getvalue()


def get_snapshot() -> dict:
    """Returns {"bytes": jpeg_bytes, "age_seconds": float, "etag": str}.
    "bytes" is None only if the camera has never been reachable at all
    (first call failed with nothing cached yet to fall back on)."""
    now = time.time()
    with _lock:
        is_fresh = _cache["bytes"] is not None and (now - _cache["fetched_at"]) < SNAPSHOT_REFRESH_SEC
        if is_fresh:
            return {**_cache, "age_seconds": now - _cache["fetched_at"]}

    try:
        raw = _fetch_from_camera()
        jpeg = _resize_and_encode(raw)
        etag = hashlib.sha1(jpeg).hexdigest()
        with _lock:
            _cache["bytes"] = jpeg
            _cache["fetched_at"] = now
            _cache["etag"] = etag
            return {**_cache, "age_seconds": 0.0}
    except Exception as exc:
        with _lock:
            if _cache["bytes"] is not None:
                print(f"[camera_snapshot] fetch failed, serving stale: {exc}")
                return {**_cache, "age_seconds": now - _cache["fetched_at"]}
            print(f"[camera_snapshot] fetch failed, no cached image yet: {exc}")
            return {"bytes": None, "age_seconds": None, "etag": None}
