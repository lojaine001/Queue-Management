"""Camera snapshots (checkout: live JPEG with ETag; entrance: last saved image)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response

import camera_snapshot

from ..settings import SNAP_DIR

router = APIRouter()


@router.get("/snapshot/checkout")
def snapshot_checkout(request: Request, format: str = None):
    if format == "json":
        # Legacy JSON form, kept for IQMSManager compatibility until it is
        # updated to the new binary endpoint too.
        import base64
        p = SNAP_DIR / "latest_checkout.jpg"
        if not p.exists():
            return {"image": None}
        with open(str(p), "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return {"image": f"data:image/jpeg;base64,{data}"}

    result = camera_snapshot.get_snapshot()
    if result["bytes"] is None:
        raise HTTPException(status_code=503, detail="Checkout camera unavailable")
    if request.headers.get("if-none-match") == result["etag"]:
        return Response(status_code=304, headers={"ETag": result["etag"]})
    return Response(
        content=result["bytes"],
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache",
            "ETag": result["etag"],
            "X-Snapshot-Age": str(round(result["age_seconds"], 1)),
        },
    )


@router.get("/snapshot/entrance")
def snapshot_entrance():
    import base64
    p = SNAP_DIR / "latest_entrance.jpg"
    if not p.exists():
        return {"image": None}
    with open(str(p), "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"image": f"data:image/jpeg;base64,{data}"}
