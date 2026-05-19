from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SIM_DIR = os.path.join(ROOT_DIR, "simulator")
APP_PATH = os.path.join(SIM_DIR, "app.py")
PYGAME_VIEWER_PATH = os.path.join(SIM_DIR, "pygame_viewer.py")
SERVER_PATH = os.path.join(SIM_DIR, "server.py")
FRONTEND_DIR = os.path.join(SIM_DIR, "frontend")
BACKEND_PORT = 8080
LOOPBACK_HOST = "127.0.0.1"


def _add_root_to_path() -> None:
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)


def _is_port_in_use(port: int) -> bool:
    endpoints = [
        (socket.AF_INET, ("127.0.0.1", port)),
        (socket.AF_INET6, ("::1", port, 0, 0)),
    ]
    for family, address in endpoints:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                if sock.connect_ex(address) == 0:
                    return True
        except OSError:
            continue
    return False


def _find_port_pids(port: int) -> list[int]:
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=False,
            )
            pids = set()
            needle = f":{port}"
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and needle in parts[1] and parts[3].upper() == "LISTENING":
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
            return sorted(pids)
    except Exception:
        pass
    return []


def _stop_port_listener(port: int) -> None:
    for pid in _find_port_pids(port):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            else:
                os.kill(pid, 15)
        except Exception:
            pass


def _terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return

    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass


def run_batch(scenario: str, calibrated: bool = False, days: int = 30) -> int:
    _add_root_to_path()
    from simulator.calibrate import calibrate
    from simulator.db import SimDB
    from simulator.engine import SimulatorEngine
    from simulator.scenarios import get_scenario

    print(f"[run_simulator] Running batch simulator for scenario: {scenario}")
    config = get_scenario(scenario)

    if calibrated:
        print(f"[run_simulator] Applying calibration from last {days} days of REAL data")
        config.update(calibrate(days=days))

    db = SimDB(scenario)
    engine = SimulatorEngine(config, db)
    engine.run()

    print(f"Simulation {scenario} complete.")
    print(f"Total people processed: {len(engine.completed_people)}")
    db.close()
    return 0


def run_smoke_test() -> int:
    _add_root_to_path()
    from simulator.test_run import test_run

    print("[run_simulator] Running simulator smoke test")
    test_run()
    return 0


def run_dashboard(calibrate: bool = False, days: int = 30) -> int:
    print("[run_simulator] Launching Streamlit dashboard")
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", APP_PATH]
    if calibrate:
        streamlit_cmd.extend(["--", "--calibrate", "--days", str(days)])

    env = os.environ.copy()
    env["SIM_EXTERNAL_SERVICES"] = "1"

    server_proc = None
    streamlit_proc = None

    try:
        _stop_port_listener(BACKEND_PORT)
        time.sleep(0.5)

        if not _is_port_in_use(BACKEND_PORT):
            server_proc = subprocess.Popen([sys.executable, SERVER_PATH], cwd=ROOT_DIR, env=env)
            time.sleep(1)

        streamlit_proc = subprocess.Popen(streamlit_cmd, cwd=ROOT_DIR, env=env)
        return streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\n[run_simulator] Ctrl+C received — stopping Streamlit dashboard and local services.")
        return 130
    finally:
        _terminate_process(streamlit_proc)
        _terminate_process(server_proc)
        _stop_port_listener(BACKEND_PORT)


def run_viewer(
    scenario: str,
    calibrate_from_db: bool = False,
    days: int = 30,
    hours: int = 8,
    speed: float = 180.0,
    fps: int = 60,
    persist: bool = False,
) -> int:
    print("[run_simulator] Launching Pygame viewer")
    cmd = [
        sys.executable,
        PYGAME_VIEWER_PATH,
        scenario,
        "--hours", str(hours),
        "--speed", str(speed),
        "--fps", str(fps),
    ]
    if calibrate_from_db:
        cmd.extend(["--calibrate", "--days", str(days)])
    if persist:
        cmd.append("--persist")
    return subprocess.call(cmd, cwd=ROOT_DIR)


def run_calibration(days: int, output: str) -> int:
    _add_root_to_path()
    from simulator.calibrate import calibrate, print_report

    print(f"[run_simulator] Calibrating simulator from last {days} days of REAL data")
    cfg = calibrate(days=days)

    if output == "json":
        print(json.dumps(cfg, indent=2, default=str))
    elif output == "py":
        print("CALIBRATED = " + json.dumps(cfg, indent=4, default=str).replace("null", "None"))
    else:
        print_report(cfg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified launcher for the IQMS queue simulator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Generate simulator data quickly and write it to PostgreSQL/TimescaleDB.",
    )
    batch_parser.add_argument(
        "scenario",
        nargs="?",
        default="normal_day",
        choices=["normal_day", "lunch_rush", "evening_rush"],
        help="Scenario preset to simulate.",
    )
    batch_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Override scenario defaults using REAL DB calibration.",
    )
    batch_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of REAL data history to use when --calibrate is enabled.",
    )

    subparsers.add_parser(
        "test",
        help="Run the simulator smoke test with a short fixed scenario.",
    )
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Open the Streamlit simulator dashboard.",
    )
    dashboard_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Preload simulator defaults from REAL DB calibration.",
    )
    dashboard_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of REAL data history to use when --calibrate is enabled.",
    )
    viewer_parser = subparsers.add_parser(
        "viewer",
        help="Open the smooth Pygame viewer for the simulator.",
    )
    viewer_parser.add_argument(
        "scenario",
        nargs="?",
        default="normal_day",
        choices=["normal_day", "lunch_rush", "evening_rush"],
        help="Scenario preset to simulate.",
    )
    viewer_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Preload simulator defaults from REAL DB calibration.",
    )
    viewer_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of REAL data history to use when --calibrate is enabled.",
    )
    viewer_parser.add_argument(
        "--hours",
        type=int,
        default=8,
        help="Simulated duration in hours.",
    )
    viewer_parser.add_argument(
        "--speed",
        type=float,
        default=180.0,
        help="Simulated seconds per real second.",
    )
    viewer_parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Viewer frames per second.",
    )
    viewer_parser.add_argument(
        "--persist",
        action="store_true",
        help="Write viewer-generated simulator rows to PostgreSQL.",
    )
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate simulator defaults from REAL PostgreSQL data.",
    )
    calibrate_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of REAL data history to analyse.",
    )
    calibrate_parser.add_argument(
        "--output",
        type=str,
        default="report",
        choices=["report", "json", "py"],
        help="Calibration output format.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "batch":
        return run_batch(args.scenario, calibrated=args.calibrate, days=args.days)
    if args.command == "test":
        return run_smoke_test()
    if args.command == "dashboard":
        return run_dashboard(calibrate=args.calibrate, days=args.days)
    if args.command == "viewer":
        return run_viewer(
            args.scenario,
            calibrate_from_db=args.calibrate,
            days=args.days,
            hours=args.hours,
            speed=args.speed,
            fps=args.fps,
            persist=args.persist,
        )
    if args.command == "calibrate":
        return run_calibration(args.days, args.output)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
