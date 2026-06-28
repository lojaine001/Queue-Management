"""
predict_runner.py — Keeps queue_predictions fresh during shop hours.

Runs ensemble_predict.py every PREDICT_INTERVAL_MIN minutes while the shop
is open.  Skips silently outside opening hours so no stale overnight rows
accumulate.

Usage
─────
    python predict_runner.py

Environment variables (all optional, inherit from core defaults)
────────────────────────────────────────────────────────────────
  SHOP_OPEN              int   Opening hour, 0-23  (default 8)
  SHOP_CLOSE             int   Closing hour, 0-23  (default 21)
  PREDICT_INTERVAL_MIN   int   Minutes between runs (default 15)
  PREDICT_SOURCE         str   REAL | SIM | ALL     (default REAL)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

_HERE        = os.path.dirname(os.path.abspath(__file__))
_PREDICT_CWD = os.path.dirname(_HERE)   # Queue-Management/ — needed for `prediction` imports
if _PREDICT_CWD not in sys.path:
    sys.path.insert(0, _PREDICT_CWD)

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(usecwd=True))

from prediction.core import is_open as _core_is_open  # respects SHOP_SCHEDULE_OVERRIDE

# ── Config ─────────────────────────────────────────────────────────────────────
PREDICT_INTERVAL_MIN = int(os.getenv("PREDICT_INTERVAL_MIN", 15))
PREDICT_SOURCE       = os.getenv("PREDICT_SOURCE", "REAL")
DATA_SPAN_DAYS       = int(os.getenv("DATA_SPAN_DAYS", 30))

_SCRIPT      = os.path.join(_HERE, "Queue-Management-System-v2-main", "ensemble_predict.py")


def _is_open() -> bool:
    return _core_is_open(pd.Timestamp.now())


def _run_once() -> bool:
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--source", PREDICT_SOURCE, "--days", str(DATA_SPAN_DAYS)],
        cwd=_PREDICT_CWD,
    )
    return result.returncode == 0


def main() -> None:
    print(f"[Runner] predict_runner started — interval {PREDICT_INTERVAL_MIN} min (schedule from SHOP_SCHEDULE_OVERRIDE)")
    print(f"[Runner] Script : {_SCRIPT}")
    print(f"[Runner] CWD    : {_PREDICT_CWD}")

    while True:
        now = datetime.now()
        if _is_open():
            print(f"[Runner] {now.strftime('%H:%M:%S')} — running prediction...")
            ok = _run_once()
            print(f"[Runner] {'OK' if ok else 'FAILED'}")
        else:
            print(f"[Runner] {now.strftime('%H:%M:%S')} — shop closed, skipping")

        time.sleep(PREDICT_INTERVAL_MIN * 60)


if __name__ == "__main__":
    main()
