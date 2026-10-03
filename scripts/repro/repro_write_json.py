"""
Reproduction script for PermissionError / WinError 5 on atomic file replacement.
Simulates a concurrent reader holding open handles without FILE_SHARE_DELETE (standard open('r') on Windows)
while write_json attempts to replace the file.
"""
import os
import sys
import time
import threading
from pathlib import Path

# Add Queue-Management and repo root to sys.path
ROOT = Path(__file__).resolve().parents[2]
QM = ROOT / "Queue-Management"
sys.path = [str(QM), str(ROOT)] + [p for p in sys.path if p not in (str(QM), str(ROOT))]

try:
    from prediction.runtime import write_json, read_json
except Exception as e:
    print(f"Could not import prediction.runtime: {e}")
    sys.exit(1)

TEST_DIR = ROOT / ".repro_test"
TEST_DIR.mkdir(parents=True, exist_ok=True)
TARGET_FILE = TEST_DIR / "scheduler_contention.json"


def reader_lock(stop_event):
    """Holds a read handle open continuously to provoke Windows file lock collision."""
    while not stop_event.is_set():
        if TARGET_FILE.exists():
            try:
                with open(TARGET_FILE, "r", encoding="utf-8") as f:
                    time.sleep(0.08)
            except Exception:
                pass
        time.sleep(0.01)


def main():
    print(f"[*] Testing write_json contention on: {TARGET_FILE}")
    # Write initial file
    write_json(TARGET_FILE, {"initial": True, "ts": time.time()})

    stop_event = threading.Event()
    readers = [threading.Thread(target=reader_lock, args=(stop_event,)) for _ in range(4)]
    for r in readers:
        r.daemon = True
        r.start()

    successes = 0
    failures = 0

    for i in range(30):
        try:
            write_json(TARGET_FILE, {"iteration": i, "ts": time.time()})
            successes += 1
            print(f"  Iteration {i:02d}: write_json OK")
        except PermissionError as e:
            failures += 1
            print(f"  Iteration {i:02d}: REPRODUCED PermissionError: {e}")
        except Exception as e:
            failures += 1
            print(f"  Iteration {i:02d}: Failed with unexpected error: {type(e).__name__}: {e}")
        time.sleep(0.02)

    stop_event.set()
    for r in readers:
        r.join(timeout=0.5)

    print("\n[Result Summary]")
    print(f"  Successes: {successes} | Failures: {failures}")
    if failures > 0:
        print("  => REPRODUCTION SUCCESSFUL: PermissionError reproduced under Windows concurrent access.")
    else:
        print("  => No failure observed with current implementation.")


if __name__ == "__main__":
    main()
