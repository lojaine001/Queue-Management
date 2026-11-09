"""Watch a checkout clip with the lane zones drawn and mark when each person arrives in and leaves a lane.

    python eval/mark_events.py --video "C:/.../checkout_20261029_143110_quiet.mkv" --start 0 --end 600

The frame is shown exactly as the pipeline sees it (resized to 640x480) with the lane polygons from
Head-Detector/config.yml. A person is "in a lane" when the bottom-centre of their head is inside the zone,
the same rule the pipeline uses.

Output: eval/data/labels/<clip>.labels.csv (+ .meta.json). Saved after every change; run the same command
again to continue where you stopped. The file holds only lanes and times, no images.

Keys (click the video window first):
  space   play / pause                  + / -   faster / slower playback
  d / a   next / previous frame         w / s   jump +5 s / -5 s
  1 2 3 4 select lane
  i       a person ARRIVES in the selected lane now
  o       the person waiting longest in the selected lane LEAVES now
  O       choose which person leaves (type their number in the terminal)
  n       add a note to a person (typed in the terminal), e.g. staff / couple / hidden 5s
  x       delete a person (typed in the terminal)
  u       undo the last change
  q       save and quit
"""
import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
HEAD_DIR = REPO_ROOT / 'Head-Detector'
FRAME_W, FRAME_H = 640, 480  # pipeline resolution (main.py resizes every frame to this)
LANE_COLOURS = [(0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255)]
CSV_FIELDS = ['clip', 'person', 'lane', 'arrive_s', 'leave_s', 'dwell_s', 'note']


def load_lanes(config_path):
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    lanes = []
    for i, roi in enumerate(cfg.get('rois', [])):
        pts = roi.get('points', [])
        if len(pts) >= 3:
            lanes.append((roi.get('name', f'lane{i + 1}'), np.array(pts, dtype=np.int32)))
    return lanes


class FrameSource:
    """Decodes the clip once into small in-memory JPEGs so any frame can be shown instantly and exactly."""

    def __init__(self, path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            sys.exit(f'Cannot open {path}')
        self.pts, self._jpgs = [], []
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            self._jpgs.append(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1])
            if len(self.pts) % 500 == 0:
                print(f'  loading... {len(self.pts)} frames ({time.time() - t0:.0f}s)')
        cap.release()
        if not self.pts:
            sys.exit('No frames decoded.')
        self.n = len(self.pts)

    def frame(self, idx):
        return cv2.imdecode(self._jpgs[idx], cv2.IMREAD_COLOR)

    def index_at(self, t):
        """Last frame at or before time t."""
        return int(np.clip(np.searchsorted(self.pts, t, side='right') - 1, 0, self.n - 1))


class Labels:
    def __init__(self, clip, csv_path, meta_path, window):
        self.clip = clip
        self.csv_path = Path(csv_path)
        self.meta_path = Path(meta_path)
        self.window = window
        self.persons = []
        self._undo = []
        if self.csv_path.exists():
            with open(self.csv_path, encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    self.persons.append({
                        'person': int(r['person']), 'lane': int(r['lane']),
                        'arrive_s': float(r['arrive_s']),
                        'leave_s': float(r['leave_s']) if r['leave_s'] else None,
                        'note': r.get('note', ''),
                    })
            print(f'Resumed {len(self.persons)} people from {self.csv_path}')

    def _snapshot(self):
        self._undo.append(copy.deepcopy(self.persons))
        del self._undo[:-50]

    def undo(self):
        if not self._undo:
            return 'nothing to undo'
        self.persons = self._undo.pop()
        self.save()
        return 'undone'

    def get(self, pid):
        return next((p for p in self.persons if p['person'] == pid), None)

    def waiting(self, lane=None, t=None):
        out = [p for p in self.persons if p['leave_s'] is None and (lane is None or p['lane'] == lane)
               and (t is None or p['arrive_s'] <= t)]
        return sorted(out, key=lambda p: p['arrive_s'])

    def arrive(self, lane, t):
        self._snapshot()
        pid = max((p['person'] for p in self.persons), default=0) + 1
        self.persons.append({'person': pid, 'lane': lane, 'arrive_s': round(t, 3), 'leave_s': None, 'note': ''})
        self.save()
        return f'#{pid} arrives in lane {lane} at {t:.1f}s'

    def leave(self, pid, t):
        p = self.get(pid)
        if p is None:
            return f'no person #{pid}'
        if t <= p['arrive_s']:
            return f'#{pid} arrived at {p["arrive_s"]:.1f}s; leave time must be later'
        self._snapshot()
        p['leave_s'] = round(t, 3)
        self.save()
        return f'#{pid} leaves lane {p["lane"]} at {t:.1f}s (stayed {t - p["arrive_s"]:.1f}s)'

    def leave_longest_waiting(self, lane, t):
        waiting = self.waiting(lane, t)
        if not waiting:
            return f'nobody waiting in lane {lane}'
        return self.leave(waiting[0]['person'], t)

    def set_note(self, pid, note):
        p = self.get(pid)
        if p is None:
            return f'no person #{pid}'
        self._snapshot()
        p['note'] = note.replace(',', ';').strip()
        self.save()
        return f'#{pid} note: {p["note"]}'

    def delete(self, pid):
        if self.get(pid) is None:
            return f'no person #{pid}'
        self._snapshot()
        self.persons = [p for p in self.persons if p['person'] != pid]
        self.save()
        return f'#{pid} deleted'

    def save(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.csv_path.with_suffix('.tmp')
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for p in sorted(self.persons, key=lambda p: p['arrive_s']):
                dwell = round(p['leave_s'] - p['arrive_s'], 3) if p['leave_s'] is not None else ''
                w.writerow({'clip': self.clip, 'person': p['person'], 'lane': p['lane'], 'arrive_s': p['arrive_s'],
                            'leave_s': '' if p['leave_s'] is None else p['leave_s'], 'dwell_s': dwell,
                            'note': p['note']})
        tmp.replace(self.csv_path)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump({'clip': self.clip, 'window_start_s': self.window[0], 'window_end_s': self.window[1],
                       'lanes_from': 'Head-Detector/config.yml', 'frame_size': [FRAME_W, FRAME_H],
                       'updated': datetime.now().isoformat(timespec='seconds')}, f, indent=2)


def draw(src, idx, lanes, labels, state, scale):
    img = src.frame(idx)
    t = src.pts[idx]
    for k, (name, pts) in enumerate(lanes, start=1):
        colour = LANE_COLOURS[(k - 1) % len(LANE_COLOURS)]
        cv2.polylines(img, [pts], True, colour, 3 if k == state['lane'] else 1, cv2.LINE_AA)
        cx, cy = pts.mean(axis=0).astype(int)
        n_wait = len(labels.waiting(k, t))
        cv2.putText(img, f'{k}' + (f' ({n_wait})' if n_wait else ''), (int(cx) - 10, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    img = cv2.resize(img, (int(FRAME_W * scale), int(FRAME_H * scale)))

    w0, w1 = labels.window
    in_window = w0 <= t <= w1
    status = 'PLAY' if state['playing'] else 'PAUSED'
    top = (f't={t:7.2f}s  frame {idx + 1}/{src.n}  {status} x{state["speed"]:g}  lane={state["lane"]}  '
           f'window {w0:g}-{w1:g}s' + ('' if in_window else '  OUTSIDE WINDOW'))
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(img, top, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255) if in_window else (0, 0, 255),
                1, cv2.LINE_AA)

    lines = [f'#{p["person"]} L{p["lane"]} since {p["arrive_s"]:.1f}s ({t - p["arrive_s"]:.0f}s)'
             for p in labels.waiting(t=t)]
    if state['msg']:
        lines.append('> ' + state['msg'])
    y = img.shape[0] - 10 - 20 * (len(lines) - 1)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (4, y - th - 4), (10 + tw, y + 4), (20, 20, 20), -1)
        cv2.putText(img, line, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return img


def ask_int(prompt):
    try:
        return int(input(prompt).strip().lstrip('#'))
    except (ValueError, EOFError):
        return None


def run_gui(src, lanes, labels, start_s, scale):
    win = 'Mark arrivals / departures  (q = save and quit)'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state = {'playing': False, 'speed': 1.0, 'lane': 1, 'msg': 'press 1-4 to pick a lane, i = arrive, o = leave'}
    idx = src.index_at(start_s)
    while True:
        cv2.imshow(win, draw(src, idx, lanes, labels, state, scale))
        if state['playing'] and idx < src.n - 1:
            delay = max(1, int((src.pts[idx + 1] - src.pts[idx]) / state['speed'] * 1000))
        else:
            delay = 30
            state['playing'] = state['playing'] and idx < src.n - 1
        key = cv2.waitKeyEx(delay)
        t = src.pts[idx]
        if key == -1:
            if state['playing']:
                idx += 1
            continue
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        ch = chr(key & 0xFF) if key & 0xFF < 128 else ''
        if ch == 'q':
            break
        elif ch == ' ':
            state['playing'] = not state['playing']
        elif ch == 'd':
            idx = min(idx + 1, src.n - 1); state['playing'] = False
        elif ch == 'a':
            idx = max(idx - 1, 0); state['playing'] = False
        elif ch == 'w':
            idx = src.index_at(t + 5)
        elif ch == 's':
            idx = src.index_at(t - 5)
        elif ch in '+=':
            state['speed'] = min(state['speed'] * 2, 16)
        elif ch == '-':
            state['speed'] = max(state['speed'] / 2, 0.25)
        elif ch in '1234' and int(ch) <= max(len(lanes), 1):
            state['lane'] = int(ch); state['msg'] = f'lane {ch} selected'
        elif ch == 'i':
            state['msg'] = labels.arrive(state['lane'], t)
        elif ch == 'o':
            state['msg'] = labels.leave_longest_waiting(state['lane'], t)
        elif ch == 'O':
            state['playing'] = False
            pid = ask_int(f'[t={t:.1f}s] number of the person who leaves: ')
            state['msg'] = labels.leave(pid, t) if pid is not None else 'cancelled'
        elif ch == 'n':
            state['playing'] = False
            pid = ask_int('note for person number: ')
            state['msg'] = labels.set_note(pid, input('note: ')) if pid is not None else 'cancelled'
        elif ch == 'x':
            state['playing'] = False
            pid = ask_int('delete person number: ')
            state['msg'] = labels.delete(pid) if pid is not None else 'cancelled'
        elif ch == 'u':
            state['msg'] = labels.undo()
        if state['msg']:
            print(state['msg'])
    labels.save()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--start', type=float, default=0.0, help='labelling window start, seconds of video')
    ap.add_argument('--end', type=float, default=600.0, help='labelling window end, seconds of video')
    ap.add_argument('--out-dir', default=str(REPO_ROOT / 'eval' / 'data' / 'labels'))
    ap.add_argument('--config', default=str(HEAD_DIR / 'config.yml'), help='lane polygons')
    ap.add_argument('--scale', type=float, default=1.5, help='display size (1.5 = 960x720)')
    args = ap.parse_args()

    video = Path(args.video)
    clip = video.stem
    lanes = load_lanes(args.config)
    if not lanes:
        sys.exit(f'No lanes (rois) found in {args.config}')
    print(f'Loading {video.name} ...')
    src = FrameSource(video)
    out = Path(args.out_dir)
    labels = Labels(clip, out / f'{clip}.labels.csv', out / f'{clip}.labels.meta.json', (args.start, args.end))
    print(__doc__.split('Keys')[1])
    run_gui(src, lanes, labels, args.start, args.scale)
    open_count = len(labels.waiting())
    print(f'Saved {len(labels.persons)} people to {labels.csv_path}'
          + (f' ({open_count} still without a leave time)' if open_count else ''))


if __name__ == '__main__':
    main()
