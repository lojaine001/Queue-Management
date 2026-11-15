"""Per-frame timing log for offline evaluation (enabled by `frame_log_enabled` in config2.yml).

The main loop only calls log(), which is a non-blocking queue put. All file I/O
happens in a daemon thread, so the live loop is never slowed down by disk writes.
If the queue is full (disk stalled), rows are dropped and counted rather than
blocking the loop; the drop count is written to the file when the logger closes.
"""
import os
import queue
import time
from datetime import datetime
from threading import Thread

_HEADER = "wall_ts,loop_sec,frame_seq,n_dets,n_tracks\n"
_STOP = object()


class FrameLogWriter:
    def __init__(self, log_dir: str, cam_id: str, max_queue: int = 5000, flush_every_sec: float = 2.0):
        self._log_dir = log_dir
        self._cam_id = cam_id
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._flush_every_sec = flush_every_sec
        self.dropped = 0
        os.makedirs(log_dir, exist_ok=True)
        self._thread = Thread(target=self._run, daemon=True, name="FrameLogWriter")
        self._thread.start()

    def log(self, wall_ts: float, loop_sec: float, frame_seq: int, n_dets: int, n_tracks: int) -> None:
        try:
            self._q.put_nowait((wall_ts, loop_sec, frame_seq, n_dets, n_tracks))
        except queue.Full:
            self.dropped += 1

    def close(self, timeout: float = 2.0) -> None:
        try:
            self._q.put(_STOP, timeout=timeout)
        except queue.Full:
            return
        self._thread.join(timeout=timeout)

    def _path_for(self, ts: float) -> str:
        day = datetime.fromtimestamp(ts).strftime('%Y%m%d')
        return os.path.join(self._log_dir, f'frame_log_{self._cam_id}_{day}.csv')

    def _run(self) -> None:
        f = None
        current_path = None
        last_flush = time.time()
        try:
            while True:
                try:
                    item = self._q.get(timeout=self._flush_every_sec)
                except queue.Empty:
                    item = None
                if item is _STOP:
                    break
                if item is not None:
                    wall_ts, loop_sec, frame_seq, n_dets, n_tracks = item
                    # One file per day, so a long-running session doesn't grow a single file forever.
                    path = self._path_for(wall_ts)
                    if path != current_path:
                        if f is not None:
                            f.close()
                        is_new = not os.path.exists(path)
                        f = open(path, 'a', encoding='utf-8', newline='')
                        if is_new:
                            f.write(_HEADER)
                        current_path = path
                    f.write(f"{wall_ts:.4f},{loop_sec:.4f},{frame_seq},{n_dets},{n_tracks}\n")
                if f is not None and time.time() - last_flush >= self._flush_every_sec:
                    f.flush()
                    last_flush = time.time()
        finally:
            if f is not None:
                if self.dropped:
                    f.write(f"# dropped_rows={self.dropped}\n")
                f.close()
