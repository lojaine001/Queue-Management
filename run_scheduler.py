"""Single-owner, observable forecast scheduler. Stops gracefully; jobs have a deadline."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import psutil
from dotenv import find_dotenv, load_dotenv
from prediction.runtime import RUNTIME, RunLock, AlreadyRunning, BUSY_EXIT, read_json, write_json
from prediction.scheduler_control import STATE_FILE, STOP_FILE, terminate_tree

HERE = Path(__file__).resolve().parent
PREDICT_SCRIPT = HERE / "Queue-Management-System-v2-main" / "Queue-Management-System-v2-main" / "ensemble_predict.py"
load_dotenv(find_dotenv(usecwd=True))
log = logging.getLogger("scheduler")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def next_deadline(previous, interval, now):
    """Skip missed slots; never create overlapping catch-up jobs."""
    return previous + (max(1, int((now - previous) // interval) + 1)) * interval


def run_job(args, state, save, stopping):
    run_id = uuid.uuid4().hex
    logs = RUNTIME / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # Keep bounded run history on disk; scheduler log uses size-based rotation.
    for old in sorted(logs.glob("*.log"), key=lambda p: p.stat().st_mtime)[:-19]:
        old.unlink(missing_ok=True)
    output = logs / (run_id + ".log")
    started = time.monotonic()
    state.update(status="running", run_id=run_id, last_start=utcnow(), active_pid=None,
                 last_error=None, log_file=str(output), next_due=None)
    save()
    QM_VENV = PREDICT_SCRIPT.parent / "venv" / "Scripts" / "python.exe"
    VENV_PY = QM_VENV if QM_VENV.exists() else (HERE.parent / ".venv" / "Scripts" / "python.exe")
    py_bin = str(VENV_PY) if VENV_PY.exists() else sys.executable
    with output.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(
            [py_bin, "-u", str(PREDICT_SCRIPT), "--source", args.source,
             "--days", str(args.days)], cwd=str(PREDICT_SCRIPT.parent),
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "PREDICTION_RUN_ID": run_id},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        state.update(active_pid=child.pid, active_created_at=psutil.Process(child.pid).create_time())
        save()
        timed_out = False
        try:
            while child.poll() is None:
                if stopping():
                    state["status"] = "stopping"
                if time.monotonic() - started >= args.timeout:
                    timed_out = True
                    terminate_tree(child.pid, state["active_created_at"])
                    child.wait(timeout=10)
                    break
                save()
                time.sleep(1)
        except BaseException:
            terminate_tree(child.pid, state["active_created_at"])
            child.wait(timeout=10)
            raise
    code = child.returncode
    state.update(active_pid=None, active_created_at=None, last_finish=utcnow(),
                 duration_seconds=round(time.monotonic()-started, 2), exit_code=code)
    if timed_out:
        state.update(status="failed", last_error=f"Forecast exceeded {args.timeout}s deadline; process tree terminated")
    elif code == 0:
        state.update(status="waiting", last_success=utcnow(), last_error=None)
    elif code == BUSY_EXIT:
        state.update(status="busy", last_error="Another forecast owns the publisher lock; this slot was skipped")
    else:
        with output.open("rb") as stream:
            stream.seek(max(0, output.stat().st_size - 2000))
            tail = stream.read().decode("utf-8", errors="replace")
        state.update(status="failed", last_error=tail[-2000:])
    save()
    log.info("Run %s finished: status=%s exit=%s duration=%ss", run_id, state["status"], code, state["duration_seconds"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=positive_int, default=os.getenv("PREDICTION_INTERVAL_MIN", "15"))
    parser.add_argument("--days", type=positive_int, default=os.getenv("DATA_SPAN_DAYS", "30"))
    parser.add_argument("--source", choices=["REAL", "SIM", "ALL"], default=os.getenv("CAM_SOURCE", "REAL"))
    parser.add_argument("--timeout", type=positive_int, default=os.getenv("PREDICTION_TIMEOUT_SEC", "1800"))
    args = parser.parse_args(argv)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(RUNTIME / "scheduler.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    with RunLock(RUNTIME / "scheduler.lock"):
        previous = read_json(STATE_FILE)
        state = dict(owner=uuid.uuid4().hex, pid=os.getpid(), created_at=psutil.Process().create_time(),
                     interval_min=args.interval, days=args.days, source=args.source, timeout_seconds=args.timeout,
                     status="starting", started_at=utcnow(), active_pid=None, last_success=previous.get("last_success"))
        def save():
            state["heartbeat"] = utcnow()
            write_json(STATE_FILE, state)
        def stopping():
            return read_json(STOP_FILE).get("owner") == state["owner"]
        save()
        due = time.monotonic()
        try:
            while not stopping():
                run_job(args, state, save, stopping)
                if stopping():
                    break
                due = next_deadline(due, args.interval * 60, time.monotonic())
                state["next_due"] = (datetime.now(timezone.utc) + timedelta(seconds=due-time.monotonic())).isoformat()
                while time.monotonic() < due and not stopping():
                    save()
                    time.sleep(min(1, max(0, due-time.monotonic())))
        except BaseException as exc:
            state.update(status="failed", last_error=f"Scheduler: {type(exc).__name__}: {exc}")
            log.exception("Scheduler failed")
            raise
        finally:
            if state["status"] != "failed":
                state["status"] = "stopped"
            state.update(active_pid=None, next_due=None, stopped_at=utcnow())
            save()


if __name__ == "__main__":
    try:
        main()
    except AlreadyRunning as exc:
        print(str(exc), flush=True)
        sys.exit(BUSY_EXIT)
