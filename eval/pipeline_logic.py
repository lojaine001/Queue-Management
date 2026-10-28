"""Offline copy of the checkout post-processing in Head-Detector/main.py.

Mirrors, in the same order, what the live loop does after tracker.update():
  - hit counting (a "hit" = the track got a new detection this frame)
  - min_delay skip, confirmation (hits / expect_fps > max_age / expect_fps)
  - lane assignment (set at confirmation, then refreshed every frame)
  - track death -> sibling suppression -> unconfirmed skip -> entrance/service event

The only intended difference is the clock: `now` is passed in (video time or the
replayed frame-log time) instead of time.time(). Drawing, equipment, snapshots and the
15-min memory pruning are left out because they don't change which events get written.

If main.py's logic changes, change this file the same way; the parity check against
production DB rows is what catches drift.
"""


class CheckoutPostProcessor:
    def __init__(self, roi_names, max_age, expect_fps, min_delay, queue_min_dwell_sec):
        self.roi_names = roi_names
        self.max_age = max_age
        self.expect_fps = expect_fps
        self.min_delay = min_delay
        self.queue_min_dwell_sec = queue_min_dwell_sec

        self.track_start_times = {}
        self.track_hits = {}
        self.track_last_det_id = {}
        self.track_roi = {}
        self.track_conf = {}
        self.confirmed = set()       # main.py: `track_id in person_loggers`
        self.prev_active_ids = set()
        self.events = []

    def _is_confirmed_by_hits(self, tid):
        return (self.track_hits.get(tid, 0) / self.expect_fps) > (self.max_age / self.expect_fps)

    def step(self, now, tracked_objects):
        """Process one frame. Returns per-track rows for this frame (for tracking metrics)."""
        rows = []
        for obj in tracked_objects:
            tid = obj.global_id

            # Hit counting, as in main.py. id() is safe here: the track still holds a
            # reference to its previous detection when the new one is compared.
            curr_det_id = id(obj.last_detection) if obj.last_detection else None
            matched = False
            if tid not in self.track_last_det_id:
                self.track_last_det_id[tid] = curr_det_id
                self.track_hits[tid] = 1 if curr_det_id else 0
                matched = curr_det_id is not None
            elif curr_det_id != self.track_last_det_id[tid] and curr_det_id is not None:
                self.track_last_det_id[tid] = curr_det_id
                self.track_hits[tid] += 1
                matched = True

            if tid not in self.track_start_times:
                self.track_start_times[tid] = now
            track_dur = now - self.track_start_times[tid]

            data = obj.last_detection.data if (hasattr(obj.last_detection, 'data') and obj.last_detection.data) else {}
            est = obj.estimate
            row = {
                'tid': tid,
                'x1': float(est[0][0]), 'y1': float(est[0][1]),
                'x2': float(est[1][0]), 'y2': float(est[1][1]),
                'matched': int(matched),
                'det_roi_idx': data.get('roi_idx', -1),
                'hits': self.track_hits[tid],
                'track_dur': track_dur,
            }

            if track_dur < self.min_delay:
                row['state'] = 'min_delay'
                rows.append(row)
                continue

            if not self._is_confirmed_by_hits(tid):
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
            self._on_death(tid, now, active_ids)
        self.prev_active_ids = active_ids
        return rows

    def _on_death(self, tid, now, active_ids):
        if tid not in self.track_start_times:
            return
        dead_roi = self.track_roi.get(tid)
        dead_dwell = now - self.track_start_times[tid]
        lane_name = self.roi_names[dead_roi] if dead_roi is not None else 'unknown'
        event = {
            'tid': tid,
            'lane_idx': -1 if dead_roi is None else dead_roi,
            'lane': lane_name,
            'entry_t': self.track_start_times[tid],
            'death_t': now,
            'dwell': dead_dwell,
            'hits': self.track_hits.get(tid, 0),
            'active_heads_in_lane': 0,
            'outcome': None,
            'written_entrance': 0,
            'written_service': 0,
        }

        if dead_roi is not None:
            siblings = [aid for aid in active_ids if self.track_roi.get(aid) == dead_roi]
            event['active_heads_in_lane'] = sum(
                1 for aid in siblings
                if aid in self.track_start_times and self._is_confirmed_by_hits(aid)
            )
            if siblings:
                sibling_dwells = [now - self.track_start_times[aid] for aid in siblings if aid in self.track_start_times]
                if (max(sibling_dwells) if sibling_dwells else 0) > dead_dwell:
                    event['outcome'] = 'suppressed_sibling'
                    self.events.append(event)
                    return

        if not self._is_confirmed_by_hits(tid):
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
            if tid not in self.track_start_times:
                continue
            roi = self.track_roi.get(tid)
            self.events.append({
                'tid': tid,
                'lane_idx': -1 if roi is None else roi,
                'lane': self.roi_names[roi] if roi is not None else 'unknown',
                'entry_t': self.track_start_times[tid],
                'death_t': now,
                'dwell': now - self.track_start_times[tid],
                'hits': self.track_hits.get(tid, 0),
                'active_heads_in_lane': 0,
                'outcome': 'censored_clip_end',
                'written_entrance': 0,
                'written_service': 0,
            })
        self.prev_active_ids = set()
