"""Cross-process ownership and atomic runtime records (no ML imports)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
BUSY_EXIT = 75

class AlreadyRunning(RuntimeError):
    pass

class RunLock:
    """OS-owned lock: automatically released even after a crash; never unlink it."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a+b")
        self.file.seek(0, 2)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise AlreadyRunning(f"Another process owns {self.path.name}") from exc
        return self

    def __exit__(self, *args):
        if self.file is not None:
            # Closing the owning handle releases the Windows / POSIX lock.
            self.file.close()
            self.file = None


def write_json(path, value):
    import random
    import stat
    import time as _time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass

    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())

        # On Windows, concurrent open handles (especially without FILE_SHARE_DELETE)
        # cause PermissionError (WinError 5) on os.replace().
        # We retry with exponential backoff and jitter.
        attempts = 8
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                os.replace(tmp, path)
                last_exc = None
                break
            except (PermissionError, OSError) as exc:
                last_exc = exc
                if attempt == attempts:
                    break
                sleep_sec = min(0.25, 0.02 * (2 ** (attempt - 1))) + random.uniform(0.005, 0.02)
                _time.sleep(sleep_sec)

        # If atomic replace persistently fails due to non-deletable sharing locks,
        # fallback to in-place write to keep the scheduler/dashboard functional.
        if last_exc is not None:
            try:
                if path.exists():
                    try:
                        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                    except OSError:
                        pass
                with open(path, "w", encoding="utf-8") as fallback_out:
                    json.dump(value, fallback_out, indent=2, allow_nan=False)
                    fallback_out.flush()
                    os.fsync(fallback_out.fileno())
            except Exception:
                raise last_exc
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass


def read_json(path):
    import time as _time
    path = Path(path)
    for attempt in range(3):
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                if attempt < 2:
                    _time.sleep(0.02)
                    continue
                return {}
            return json.loads(text)
        except (OSError, ValueError):
            if attempt < 2:
                _time.sleep(0.02)
                continue
            return {}
    return {}
