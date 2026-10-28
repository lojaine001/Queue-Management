"""Offline copy of the checkout post-processing in Head-Detector/main.py.

Mirrors, in the same order, what the live loop does after tracker.update():
  - frame_clock.tick() and the confirmation window frames_to_sec(max_age)
  - hit detection + track_clock.update() (the production FrameClock/TrackClock, imported, not copied)
  - min_delay skip, confirmation (track_clock.hits > min_confirmed_track_sec)
  - lane assignment (set at confirmation, then refreshed every frame)
  - track death -> sibling suppression -> unconfirmed skip -> entrance/service event

The only intended difference is the clock: `now` is passed in (video time or the
replayed frame-log time) instead of time.time()/perf_counter(). Drawing, equipment,
snapshots and the 15-min memory pruning are left out because they don't change which
events get written.

If main.py's logic changes, change this file the same way; the parity check against
production DB rows is what catches drift. Requires Head-Detector on sys.path.
"""
from utils.track_clock import FrameClock, TrackClock


class CheckoutPostProcessor:
    def __init__(self, roi_names, max_age, min_delay, queue_min_dwell_sec):
        self.roi_names = roi_names
        self.max_age = max_age
        self.min_delay = min_delay
        self.queue_min_dwell_sec = queue_min_dwell_sec

        self.frame_clock = FrameClock()
        self.track_clock = TrackClock()
        self.track_start_times = {}
        self.track_last_det_id = {}
        self.track_n_hits = {}       # eval only: matched-frame count, for the outputs
        self.track_roi = {}
        self.track_conf = {}
        self.confirmed = set()       # main.py: `track_id in person_loggers`
        self.prev_active_ids = set()
        self.events = []

    def step(self, now, tracked_objects):
        """Process one frame. Returns per-track rows for this frame (for tracking metrics)."""
        self.frame_clock.tick(now)
        min_confirmed_track_sec = self.frame_clock.frames_to_sec(self.max_age)

        rows = []
        for obj in tracked_objects:
            tid = obj.global_id

            # Same hit rule as main.py. id() is safe here: the track still holds a
            # reference to its previous detection when the new one is compared.
            curr_det_id = id(obj.last_detection) if obj.last_detection else None
            first_seen = tid not in self.track_last_det_id
            hit = curr_det_id is not None and (first_seen or curr_det_id != self.track_last_det_id[tid])
            if first_seen or hit:
                self.track_last_det_id[tid] = curr_det_id
            self.track_clock.update(tid, obj.age, self.frame_clock, hit)
            self.track_n_hits[tid] = self.track_n_hits.get(tid, 0) + int(hit)

            if tid not in self.track_start_times:
                self.track_start_times[tid] = now
            track_dur = now - self.track_start_times[tid]

            data = obj.last_detection.data if (hasattr(obj.last_detection, 'data') and obj.last_detection.data) else {}
            est = obj.estimate
            row = {
                'tid': tid,
                'x1': float(est[0][0]), 'y1': float(est[0][1]),
                'x2': float(est[1][0]), 'y2': float(est[1][1]),
                'matched': int(hit),
                'det_roi_idx': data.get('roi_idx', -1),
                'n_hits': self.track_n_hits[tid],
                'hit_sec': self.track_clock.hits(tid),
                'track_dur': track_dur,
            }

            if track_dur < self.min_delay:
                row['state'] = 'min_delay'
                rows.append(row)
                continue

            if self.track_clock.hits(tid) <= min_confirmed_track_sec:
                row['state'] = 'tentative'
                rows.append(row)
                continue

            if tid not in self.confirmed:
                self.track_roi[tid] = data.get('roi_idx')
                self.track_conf[tid] = data.get('confidence', 0.0)
                self.confirmed.add(tid)

            current_roi_idx = data.get('roi_idx')
            if current_roi_idx is not None:
                self.track_roi[tid] = current_roi_idx

            row['state'] = 'confirmed'
            rows.append(row)

        active_ids = {obj.global_id for obj in tracked_objects}
        for tid in sorted(self.prev_active_ids - active_ids):
            self._on_death(tid, now, active_ids, min_confirmed_track_sec)
        self.prev_active_ids = active_ids
        return rows

    def _event(self, tid, roi, now, outcome):
        return {
            'tid': tid,
            'lane_idx': -1 if roi is None else roi,
            'lane': self.roi_names[roi] if roi is not None else 'unknown',
            'entry_t': self.track_start_times[tid],
            'death_t': now,
            'dwell': now - self.track_start_times[tid],
            'n_hits': self.track_n_hits.get(tid, 0),
            'hit_sec': self.track_clock.hits(tid),
            'active_heads_in_lane': 0,
            'outcome': outcome,
            'written_entrance': 0,
            'written_service': 0,
        }

    def _on_death(self, tid, now, active_ids, min_confirmed_track_sec):
        if tid not in self.track_start_times:
            return
        dead_roi = self.track_roi.get(tid)
        dead_dwell = now - self.track_start_times[tid]
        event = self._event(tid, dead_roi, now, None)

        if dead_roi is not None:
            siblings = [aid for aid in active_ids if self.track_roi.get(aid) == dead_roi]
            event['active_heads_in_lane'] = sum(
                1 for aid in siblings
                if aid in self.track_start_times and self.track_clock.hits(aid) > min_confirmed_track_sec
            )
            if siblings:
                sibling_dwells = [now - self.track_start_times[aid] for aid in siblings if aid in self.track_start_times]
                if (max(sibling_dwells) if sibling_dwells else 0) > dead_dwell:
                    event['outcome'] = 'suppressed_sibling'
                    self.events.append(event)
                    return

        if self.track_clock.hits(tid) <= min_confirmed_track_sec:
            event['outcome'] = 'skipped_unconfirmed'
            self.events.append(event)
            return

        event['outcome'] = 'inserted'
        event['written_entrance'] = 1
        event['written_service'] = int(dead_dwell >= self.queue_min_dwell_sec)
        self.events.append(event)

    def finish(self, now):
        """Tracks still alive when the clip ends: not in the DB yet, flagged so evaluation can exclude them."""
        for tid in sorted(self.prev_active_ids):
            if tid in self.track_start_times:
                self.events.append(self._event(tid, self.track_roi.get(tid), now, 'censored_clip_end'))
        self.prev_active_ids = set()
