"""Dashboard-side scheduler control without PID-only ownership assumptions."""
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import time
import psutil
from prediction.runtime import RUNTIME, ROOT, read_json, write_json

STATE_FILE = RUNTIME / "scheduler.json"
STOP_FILE = RUNTIME / "stop.json"


def owned_process(pid, created_at):
    try:
        proc = psutil.Process(int(pid))
        if abs(proc.create_time() - float(created_at)) < 0.01 and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            return proc
    except (psutil.Error, ValueError, TypeError):
        pass
    return None


def scheduler_state():
    state = read_json(STATE_FILE)
    proc = owned_process(state.get("pid"), state.get("created_at"))
    try:
        live = proc is not None and any(Path(arg).name == "run_scheduler.py" for arg in proc.cmdline())
    except psutil.Error:
        live = False
    state["alive"] = live
    if not live:
        state["status"] = "failed" if state.get("status") not in (None, "stopped") else "stopped"
    else:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(state["heartbeat"])).total_seconds()
        except (KeyError, ValueError):
            age = float("inf")
        if age > 15:
            state["status"] = "unresponsive"
        elif read_json(STOP_FILE).get("owner") == state.get("owner"):
            state["status"] = "stopping"
    return state


def request_stop():
    state = scheduler_state()
    if not state.get("alive"):
        return False
    write_json(STOP_FILE, {"owner": state["owner"], "requested_at": datetime.now(timezone.utc).isoformat()})
    return True


def terminate_tree(pid, created_at):
    """Kill and verify the exact owned tree. Used only for deadlines / cancellation."""
    proc = owned_process(pid, created_at)
    if proc is None:
        return
    children = proc.children(recursive=True)
    # Stop spawning first; snapshot all descendants before terminating the parent.
    for item in [proc] + children:
        try:
            item.suspend()
        except psutil.NoSuchProcess:
            pass
    for item in reversed(children + [proc]):
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children + [proc], timeout=10)
    if alive:
        raise RuntimeError(f"Processes did not stop: {[p.pid for p in alive]}")


def start_scheduler(interval_min, days):
    state = scheduler_state()
    if state.get("alive"):
        return state["pid"]
    RUNTIME.mkdir(parents=True, exist_ok=True)
    VENV_PY = ROOT.parent / ".venv" / "Scripts" / "python.exe"
    py_bin = str(VENV_PY) if VENV_PY.exists() else sys.executable
    with (RUNTIME / "startup.log").open("a", encoding="utf-8") as output:
        child = subprocess.Popen(
            [py_bin, "-u", str(ROOT / "run_scheduler.py"), "--interval", str(interval_min), "--days", str(days)],
            cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(50):
        state = scheduler_state()
        if state.get("alive"):
            return state["pid"]
        if child.poll() is not None:
            raise RuntimeError("Scheduler could not start; see .runtime/startup.log")
        time.sleep(0.1)
    raise RuntimeError("Scheduler startup did not report healthy state within 5 seconds")


def run_hidden(command, *, cwd, env, timeout=1800):
    """Bound a manual job and all descendants without opening a Windows console."""
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    created = psutil.Process(proc.pid).create_time()
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_tree(proc.pid, created)
        out, err = proc.communicate(timeout=10)
        return subprocess.CompletedProcess(command, 124, out, err + f"\nForecast exceeded {timeout}s deadline; process tree stopped.")
    return subprocess.CompletedProcess(command, proc.returncode, out, err)
