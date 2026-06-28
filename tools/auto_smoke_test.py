#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
LOG_DIR = ROOT / "tools" / "smoke_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

HYBRID_APP = ROOT / "simulator" / "app_hybrid_wait.py"
OLD_DASHBOARD = ROOT / "Queue-Management-System-v2-main" / "Queue-Management-System-v2-main" / "dashboard.py"
PIPELINE = ROOT / "prediction" / "pipeline.py"
HYBRID_WAIT = ROOT / "prediction" / "hybrid_wait.py"
ENSEMBLE = ROOT / "Queue-Management-System-v2-main" / "Queue-Management-System-v2-main" / "ensemble_predict.py"
BACKTEST = ROOT / "Queue-Management-System-v2-main" / "Queue-Management-System-v2-main" / "backtest_predict.py"
ROOT_VENV_PY = ROOT.parent / ".venv" / "Scripts" / "python.exe"
SIM_VENV_PY = ROOT / "simulator" / ".venv" / "Scripts" / "python.exe"
OLD_VENV_PY = ROOT / "Queue-Management-System-v2-main" / "Queue-Management-System-v2-main" / "venv" / "Scripts" / "python.exe"


@dataclass
class SmokeResult:
    name: str
    ok: bool
    kind: str
    started_at: float
    ended_at: float
    duration_sec: float
    detail: str
    log_path: str | None = None
    exit_code: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


class LineBuffer:
    def __init__(self, log_file):
        self.lines: list[str] = []
        self._log_file = log_file
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            self._log_file.write(line)
            self._log_file.flush()

    def joined(self) -> str:
        with self._lock:
            return "".join(self.lines)

    def contains(self, needle: str) -> bool:
        with self._lock:
            return any(needle in line for line in self.lines)


def _now() -> float:
    return time.time()


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _pick_python(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return PYTHON


def _windows_path(path: Path | str) -> str:
    raw = str(path)
    if raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5].upper()
        tail = raw[6:].replace("/", "\\")
        return f"{drive}:{tail}"
    return raw


def _path_for_exec(exec_path: str, path: Path | str) -> str:
    if exec_path.lower().endswith(".exe"):
        return _windows_path(path)
    return str(path)


def _reader_thread(pipe, buffer: LineBuffer):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            buffer.append(line)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _run_cmd(
    *,
    name: str,
    cmd: Sequence[str],
    timeout_sec: int = 180,
    expect_substrings: Sequence[str] = (),
    allow_nonzero: bool = False,
    stop_on_markers: bool = False,
) -> SmokeResult:
    started = _now()
    log_path = LOG_DIR / f"{_timestamp()}_{name.replace(' ', '_')}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=_base_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        buffer = LineBuffer(log_file)
        t = threading.Thread(target=_reader_thread, args=(proc.stdout, buffer), daemon=True)
        t.start()
        timed_out = False
        early_success = False
        deadline = time.time() + timeout_sec
        exit_code: int | None = None
        while time.time() < deadline:
            exit_code = proc.poll()
            output = buffer.joined()
            missing = [s for s in expect_substrings if s not in output]
            if stop_on_markers and not missing:
                early_success = True
                break
            if exit_code is not None:
                break
            time.sleep(0.5)
        if early_success and proc.poll() is None:
            proc.kill()
            exit_code = proc.wait()
        elif exit_code is None:
            timed_out = True
            proc.kill()
            exit_code = proc.wait()
        t.join(timeout=2)
        output = buffer.joined()
        missing = [s for s in expect_substrings if s not in output]
        ok = (not timed_out) and (early_success or allow_nonzero or exit_code == 0) and not missing
        detail = "ok"
        if timed_out:
            detail = f"timed out after {timeout_sec}s"
        elif not early_success and exit_code != 0 and not allow_nonzero:
            detail = f"exit code {exit_code}"
        elif missing:
            detail = f"missing expected log markers: {', '.join(missing)}"
        ended = _now()
        return SmokeResult(
            name=name,
            ok=ok,
            kind="command",
            started_at=started,
            ended_at=ended,
            duration_sec=ended - started,
            detail=detail,
            log_path=str(log_path),
            exit_code=exit_code,
        )


def _http_get(url: str, timeout: int = 5) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            code = getattr(resp, "status", 200)
            body = resp.read(512).decode("utf-8", errors="ignore")
            return (200 <= code < 400), f"http {code} {body[:120]}"
    except urllib.error.HTTPError as exc:
        return False, f"http error {exc.code}"
    except Exception as exc:
        return False, str(exc)


def _terminate_proc(proc: subprocess.Popen[str]) -> int | None:
    if proc.poll() is not None:
        return proc.returncode
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
    return proc.returncode


def _pkill_pattern(pattern: str) -> None:
    try:
        subprocess.run(["pkill", "-f", pattern], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _run_streamlit_smoke(
    *,
    name: str,
    script_path: Path,
    port: int,
    python_exec: str,
    ready_markers: Sequence[str] = (),
    timeout_sec: int = 180,
) -> SmokeResult:
    started = _now()
    log_path = LOG_DIR / f"{_timestamp()}_{name.replace(' ', '_')}.log"
    _pkill_pattern(script_path.name)
    cmd = [
        python_exec,
        "-m",
        "streamlit",
        "run",
        _path_for_exec(python_exec, script_path),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=_base_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=(os.name != "nt"),
        )
        buffer = LineBuffer(log_file)
        t = threading.Thread(target=_reader_thread, args=(proc.stdout, buffer), daemon=True)
        t.start()
        root_url = f"http://127.0.0.1:{port}/"
        health_url = f"http://127.0.0.1:{port}/_stcore/health"
        page_ready = False
        health_ready = False
        startup_ready = False
        marker_ready = not ready_markers
        last_http = ""
        deadline = time.time() + timeout_sec
        while time.time() < deadline and proc.poll() is None:
            startup_ready = startup_ready or buffer.contains("You can now view your Streamlit app")
            ok_root, root_msg = _http_get(root_url, timeout=2)
            ok_health, health_msg = _http_get(health_url, timeout=2)
            page_ready = page_ready or ok_root
            health_ready = health_ready or ok_health
            last_http = f"root={root_msg}; health={health_msg}"
            if not marker_ready:
                marker_ready = all(buffer.contains(marker) for marker in ready_markers)
            if startup_ready and marker_ready and (page_ready or health_ready or (time.time() - started) >= 3.0):
                break
            time.sleep(1)

        ok = startup_ready and marker_ready and proc.poll() is None and (page_ready or health_ready or startup_ready)
        detail = "ok"
        if not ok:
            if proc.poll() is not None:
                detail = f"streamlit exited early with code {proc.returncode}"
            elif not startup_ready:
                detail = f"streamlit startup banner not seen within {timeout_sec}s"
            elif not page_ready and not health_ready:
                detail = f"page not ready within {timeout_sec}s ({last_http})"
            else:
                detail = f"missing ready markers: {', '.join([m for m in ready_markers if not buffer.contains(m)])}"

        extra: dict[str, object] = {
            "root_url": root_url,
            "health_url": health_url,
            "startup_ready": startup_ready,
            "page_ready": page_ready,
            "health_ready": health_ready,
            "marker_ready": marker_ready,
        }

        if ok and (page_ready or health_ready):
            refresh_ok_1, refresh_msg_1 = _http_get(root_url + f"?smoke=1&t={int(time.time())}", timeout=5)
            time.sleep(1)
            refresh_ok_2, refresh_msg_2 = _http_get(root_url + f"?smoke=2&t={int(time.time())}", timeout=5)
            extra["refresh_1"] = refresh_msg_1
            extra["refresh_2"] = refresh_msg_2
            extra["refresh_ok"] = bool(refresh_ok_1 and refresh_ok_2)
            if not (refresh_ok_1 and refresh_ok_2):
                ok = False
                detail = f"page refresh check failed ({refresh_msg_1}; {refresh_msg_2})"

        exit_code = _terminate_proc(proc)
        _pkill_pattern(script_path.name)
        _pkill_pattern(f"streamlit run {_path_for_exec(python_exec, script_path)}")
        t.join(timeout=2)
        ended = _now()
        return SmokeResult(
            name=name,
            ok=ok,
            kind="streamlit",
            started_at=started,
            ended_at=ended,
            duration_sec=ended - started,
            detail=detail,
            log_path=str(log_path),
            exit_code=exit_code,
            extra=extra,
        )


def _write_report(results: list[SmokeResult]) -> Path:
    report_path = LOG_DIR / f"{_timestamp()}_smoke_report.json"
    data = [asdict(r) for r in results]
    report_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return report_path


def _print_summary(results: list[SmokeResult], report_path: Path) -> int:
    print("\nSmoke summary\n")
    failures = 0
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name} ({result.kind}, {result.duration_sec:.1f}s)")
        print(f"  detail: {result.detail}")
        if result.log_path:
            print(f"  log: {result.log_path}")
        if result.extra:
            print(f"  extra: {result.extra}")
        if not result.ok:
            failures += 1
    print(f"\nReport: {report_path}")
    print(f"Passed: {len(results) - failures}/{len(results)}")
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated smoke runner for queue/wait prediction apps and scripts.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional subset of test ids: compile pipeline hybrid ensemble backtest hybrid_app dashboard_app",
    )
    args = parser.parse_args()
    only = set(args.only)

    results: list[SmokeResult] = []

    def enabled(test_id: str) -> bool:
        return not only or test_id in only

    if enabled("compile"):
        results.append(
            _run_cmd(
                name="compile targets",
                cmd=[
                    PYTHON,
                    "-m",
                    "py_compile",
                    str(PIPELINE),
                    str(HYBRID_WAIT),
                    str(HYBRID_APP),
                    str(OLD_DASHBOARD),
                    str(ENSEMBLE),
                    str(BACKTEST),
                ],
                timeout_sec=60,
            )
        )

    if enabled("pipeline"):
        code = """
from prediction import pipeline
pipeline.save_prediction = lambda **kwargs: print("[SMOKE] save_prediction skipped")
result = pipeline.run_prediction_pipeline(source="REAL", days=7, use_bootstrap=False)
print("[SMOKE] pipeline ok", sorted([k for k in ["wait_15m","wait_30m","forecast","current_queue","active_lanes"] if k in result]))
print("[SMOKE] forecast rows", len(result.get("forecast", [])))
"""
        pipeline_python = _pick_python(OLD_VENV_PY, SIM_VENV_PY, ROOT_VENV_PY)
        results.append(
            _run_cmd(
                name="pipeline dry run",
                cmd=[pipeline_python, "-c", code],
                timeout_sec=240,
                expect_substrings=["[SMOKE] pipeline ok"],
                stop_on_markers=True,
            )
        )

    if enabled("hybrid"):
        code = """
from prediction import hybrid_wait
base = hybrid_wait.load_base_real_prediction(days=7, use_bootstrap=False)
view = hybrid_wait.build_hybrid_wait_view(base, current_weight_pct=70, auto_shift_history=True, recent_window_min=45, lane_strategy="auto")
print("[SMOKE] hybrid ok", view.get("wait_15m"), view.get("wait_30m"), view.get("active_lanes_now"))
"""
        hybrid_python = _pick_python(OLD_VENV_PY, SIM_VENV_PY, ROOT_VENV_PY)
        results.append(
            _run_cmd(
                name="hybrid dry run",
                cmd=[hybrid_python, "-c", code],
                timeout_sec=240,
                expect_substrings=["[SMOKE] hybrid ok"],
                stop_on_markers=True,
            )
        )

    if enabled("ensemble"):
        ensemble_python = _pick_python(OLD_VENV_PY, ROOT_VENV_PY)
        ensemble_path = _path_for_exec(ensemble_python, ENSEMBLE)
        code = f"""
import importlib.util
import pickle
import psycopg2
import tensorflow as tf
import xgboost as xgb
spec = importlib.util.spec_from_file_location("ensemble_predict", r"{ensemble_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
conn = psycopg2.connect(**mod.DB_CONFIG)
cur = conn.cursor()
cur.execute("SELECT 1")
cur.fetchone()
cur.close()
conn.close()
with open(mod.PROPHET_PATH, "rb") as fh:
    pickle.load(fh)
tf.keras.models.load_model(mod.LSTM_PATH)
with open(mod.SCALER_PATH, "rb") as fh:
    pickle.load(fh)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(mod.XGB_PATH)
print("[SMOKE] ensemble ok", mod.PROPHET_PATH, mod.LSTM_PATH, mod.XGB_PATH)
"""
        results.append(
            _run_cmd(
                name="ensemble dry run",
                cmd=[ensemble_python, "-c", code],
                timeout_sec=300,
                expect_substrings=["[SMOKE] ensemble ok"],
                stop_on_markers=True,
            )
        )

    if enabled("backtest"):
        backtest_python = _pick_python(OLD_VENV_PY, ROOT_VENV_PY)
        backtest_path = _path_for_exec(backtest_python, BACKTEST)
        code = f"""
import importlib.util, pickle
import tensorflow as tf
import xgboost as xgb
spec = importlib.util.spec_from_file_location("backtest_predict", r"{backtest_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod._check_models()
tf.keras.models.load_model(mod.LSTM_PATH)
with open(mod.SCALER_PATH, "rb") as fh:
    pickle.load(fh)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(mod.XGB_PATH)
print("[SMOKE] backtest models ok", mod.LSTM_PATH, mod.SCALER_PATH, mod.XGB_PATH)
"""
        results.append(
            _run_cmd(
                name="backtest model readiness",
                cmd=[backtest_python, "-c", code],
                timeout_sec=180,
                expect_substrings=["[SMOKE] backtest models ok"],
                stop_on_markers=True,
            )
        )

    if enabled("hybrid_app"):
        results.append(
            _run_streamlit_smoke(
                name="hybrid app",
                script_path=HYBRID_APP,
                port=8511,
                python_exec=_pick_python(OLD_VENV_PY, SIM_VENV_PY, ROOT_VENV_PY),
                ready_markers=[],
                timeout_sec=240,
            )
        )

    if enabled("dashboard_app"):
        results.append(
            _run_streamlit_smoke(
                name="dashboard app",
                script_path=OLD_DASHBOARD,
                port=8512,
                python_exec=_pick_python(OLD_VENV_PY, ROOT_VENV_PY),
                ready_markers=[],
                timeout_sec=240,
            )
        )

    report_path = _write_report(results)
    return _print_summary(results, report_path)


if __name__ == "__main__":
    raise SystemExit(main())
