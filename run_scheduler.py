"""
run_scheduler.py

Runs ensemble_predict.py automatically every PREDICTION_INTERVAL_MIN minutes.
Start this once and leave it running alongside the dashboard.

Configuration (env vars or .env):
    PREDICTION_INTERVAL_MIN   int   Minutes between prediction runs (default: 15)
    DATA_SPAN_DAYS            int   Training lookback in days passed to --days (default: 30)
    CAM_SOURCE                str   Source filter: REAL / SIM / ALL (default: REAL)

Usage:
    python run_scheduler.py
    python run_scheduler.py --interval 10 --days 14
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

HERE          = Path(__file__).resolve().parent
PREDICT_SCRIPT = HERE / "Queue-Management-System-v2-main" / "Queue-Management-System-v2-main" / "ensemble_predict.py"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_once(source: str, days: int) -> bool:
    print(f"\n[{_now()}] Running prediction (source={source}, days={days})...")
    result = subprocess.run(
        [sys.executable, str(PREDICT_SCRIPT), "--source", source, "--days", str(days)],
        cwd=str(HERE),
    )
    ok = result.returncode == 0
    status = "✓ complete" if ok else f"✗ FAILED (exit {result.returncode})"
    print(f"[{_now()}] Prediction {status}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Auto-run prediction every N minutes.")
    parser.add_argument("--interval", type=int,
                        default=int(os.getenv("PREDICTION_INTERVAL_MIN", 15)),
                        help="Minutes between runs (default: 15)")
    parser.add_argument("--days", type=int,
                        default=int(os.getenv("DATA_SPAN_DAYS", 7)),
                        help="Training data span in days (default: 30)")
    parser.add_argument("--source", type=str,
                        default=os.getenv("CAM_SOURCE", "REAL"),
                        choices=["REAL", "SIM", "ALL"])
    args = parser.parse_args()

    if not PREDICT_SCRIPT.exists():
        print(f"[Scheduler] ERROR: prediction script not found at {PREDICT_SCRIPT}")
        sys.exit(1)

    print(f"[Scheduler] Starting — every {args.interval} min | source={args.source} | days={args.days}")
    print(f"[Scheduler] Script: {PREDICT_SCRIPT}")
    print(f"[Scheduler] Press Ctrl+C to stop.\n")

    run_once(args.source, args.days)

    while True:
        next_run = time.time() + args.interval * 60
        remaining = args.interval * 60
        while remaining > 0:
            mins, secs = divmod(int(remaining), 60)
            print(f"\r[{_now()}] Next run in {mins:02d}:{secs:02d}...", end="", flush=True)
            time.sleep(min(10, remaining))
            remaining = next_run - time.time()
        print()
        run_once(args.source, args.days)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[{_now()}] Scheduler stopped.")
