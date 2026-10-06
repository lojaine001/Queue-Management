"""
Measures real-world bandwidth cost of the /snapshot/checkout endpoint
(IQMSManager Live screen), ahead of any caching/resizing work (Feature 4).

Calls GET /snapshot/checkout N times against a running api.py, and reports:
  - average JSON response body size (base64 data: URI, what actually goes
    over the wire to the app)
  - average decoded JPEG size
  - image resolution (assumed constant across calls; reported if it varies)
  - projected MB/hour for one open client, given the Live screen's own
    polling interval (currently 5000ms, see IQMSManager/App.js:328)

Usage:
    python tools/measure_snapshot_usage.py [--url http://localhost:8000] [--calls 20] [--interval-s 5]
"""
from __future__ import annotations

import argparse
import base64
import io
import statistics
import time

import requests

try:
    from PIL import Image
except ImportError:
    Image = None


def measure(base_url: str, calls: int) -> dict:
    endpoint = f"{base_url.rstrip('/')}/snapshot/checkout"
    json_bytes_samples = []
    jpeg_bytes_samples = []
    resolutions = set()
    failures = 0

    for i in range(calls):
        resp = requests.get(endpoint, timeout=10)
        resp.raise_for_status()
        body = resp.content
        json_bytes_samples.append(len(body))

        payload = resp.json()
        data_uri = payload.get("image")
        if not data_uri:
            failures += 1
            continue

        _, _, b64 = data_uri.partition(",")
        jpeg_bytes = base64.b64decode(b64)
        jpeg_bytes_samples.append(len(jpeg_bytes))

        if Image is not None:
            with Image.open(io.BytesIO(jpeg_bytes)) as im:
                resolutions.add(im.size)

    if not json_bytes_samples:
        raise RuntimeError("No successful responses -- is the API running and reachable?")

    return {
        "calls": calls,
        "failures": failures,
        "avg_json_bytes": statistics.mean(json_bytes_samples),
        "avg_jpeg_bytes": statistics.mean(jpeg_bytes_samples) if jpeg_bytes_samples else None,
        "resolutions": resolutions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000", help="Base URL of a running api.py")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--interval-s", type=float, default=5.0,
                     help="Live screen's own polling interval -- IQMSManager/App.js:328 (5000ms)")
    args = ap.parse_args()

    result = measure(args.url, args.calls)

    mb_per_hour = result["avg_json_bytes"] * (3600 / args.interval_s) / 1e6

    print(f"Endpoint:            GET {args.url}/snapshot/checkout")
    print(f"Calls:               {result['calls']} ({result['failures']} returned no image)")
    print(f"Avg JSON body size:  {result['avg_json_bytes']:.0f} bytes")
    if result["avg_jpeg_bytes"] is not None:
        print(f"Avg decoded JPEG:    {result['avg_jpeg_bytes']:.0f} bytes")
    else:
        print("Avg decoded JPEG:    n/a (Pillow not installed -- pip install pillow)")
    print(f"Resolution(s) seen:  {sorted(result['resolutions']) or 'n/a (Pillow not installed)'}")
    print(f"Polling interval:    {args.interval_s:.0f}s (Live screen, checkout + entrance polled together)")
    print(f"--> {mb_per_hour:.2f} MB/hour per open client, for /snapshot/checkout alone")
    print("    (the Live screen polls /snapshot/entrance on the same 5s interval --")
    print("     real per-client bandwidth for both images is roughly 2x this figure)")


if __name__ == "__main__":
    main()
