import subprocess
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / 'Queue-Management' / 'Queue-Management-System-v2-main' / 'Queue-Management-System-v2-main'
STDOUT = APP_DIR / 'streamlit_stdout.log'
STDERR = APP_DIR / 'streamlit_stderr.log'
VENV_PY = ROOT / '.venv' / 'Scripts' / 'python.exe'
PY = str(VENV_PY) if VENV_PY.exists() else r"C:\Program Files\Python310\python.exe"

def stop_existing(port=8501):
    # attempts to find a python process listening on port and kill it
    try:
        import psutil
    except Exception:
        print('psutil not available; please stop streamlit manually')
        return
    procs_to_stop = []
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr and conn.laddr.port == port and conn.pid:
            try:
                p = psutil.Process(conn.pid)
                if p not in procs_to_stop:
                    procs_to_stop.append(p)
            except Exception:
                pass
    for p in procs_to_stop:
        try:
            print('stopping', p.pid, p.name())
            p.terminate()
        except Exception as e:
            print('stop failed', e)
    if procs_to_stop:
        _, alive = psutil.wait_procs(procs_to_stop, timeout=5)
        for p in alive:
            try:
                p.kill()
            except Exception:
                pass
        time.sleep(1)


def start():
    cmd = [PY, '-u', '-m', 'streamlit', 'run', 'dashboard.py', '--server.port=8501']
    print('starting:', cmd, 'cwd=', APP_DIR)
    out = STDOUT.open('wb')
    err = STDERR.open('wb')
    p = subprocess.Popen(cmd, cwd=str(APP_DIR), stdout=out, stderr=err, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    time.sleep(3)
    print('started pid', p.pid)


if __name__ == '__main__':
    stop_existing()
    start()
