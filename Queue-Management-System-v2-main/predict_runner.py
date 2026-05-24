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

# ── Config ─────────────────────────────────────────────────────────────────────
SHOP_OPEN            = int(os.getenv("SHOP_OPEN",         8))
SHOP_OPEN_MINUTE     = int(os.getenv("SHOP_OPEN_MINUTE",  30))
SHOP_CLOSE_HOUR      = int(os.getenv("SHOP_CLOSE_HOUR",  20))
SHOP_CLOSE_MINUTE    = int(os.getenv("SHOP_CLOSE_MINUTE", 30))
SHOP_OPEN_TOT        = SHOP_OPEN * 60 + SHOP_OPEN_MINUTE       # 510
SHOP_CLOSE_TOT       = SHOP_CLOSE_HOUR * 60 + SHOP_CLOSE_MINUTE  # 1230
PREDICT_INTERVAL_MIN = int(os.getenv("PREDICT_INTERVAL_MIN", 15))
PREDICT_SOURCE       = os.getenv("PREDICT_SOURCE", "REAL")

_HERE        = os.path.dirname(os.path.abspath(__file__))
_SCRIPT      = os.path.join(_HERE, "Queue-Management-System-v2-main", "ensemble_predict.py")
_PREDICT_CWD = os.path.dirname(_HERE)   # Queue-Management/ — needed for `prediction` imports


def _is_open() -> bool:
    now = datetime.now()
    tot = now.hour * 60 + now.minute
    return SHOP_OPEN_TOT <= tot < SHOP_CLOSE_TOT


def _run_once() -> bool:
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--source", PREDICT_SOURCE],
        cwd=_PREDICT_CWD,
    )
    return result.returncode == 0


def main() -> None:
    print(f"[Runner] predict_runner started — interval {PREDICT_INTERVAL_MIN} min, "
          f"shop {SHOP_OPEN:02d}:{SHOP_OPEN_MINUTE:02d}–{SHOP_CLOSE_HOUR:02d}:{SHOP_CLOSE_MINUTE:02d}")
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
