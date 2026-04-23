import heapq
import random
from datetime import datetime, timedelta
from typing import Dict, Any
from simulator.entities import Person, Lane
from simulator.db import SimDB

class SimulatorEngine:
    def __init__(self, scenario_config: Dict[str, Any], db: SimDB):
        self.config = scenario_config
        self.db = db
        self.open_hour = int(scenario_config.get('open_hour', 8))
        self.close_hour = int(scenario_config.get('close_hour', 20))

        start_time = scenario_config.get('start_time', datetime.now())
        self.current_time = self._next_open_time(start_time)
        self.end_time = scenario_config.get('end_time', self.current_time + timedelta(hours=8))
        
        self.event_queue = []
        self.lanes = [Lane(lane_id=i, camera_id=db.camera_id) for i in range(scenario_config.get('num_lanes', 2))]
        self.people_at_entrance = []
        self.people_in_shop = []
        self.people_in_queue = []
        self.completed_people = []
        self.snapshot_interval_seconds = int(scenario_config.get('snapshot_interval_seconds', 300))
        
        self.person_id_counter = 1
        
        # Schedule first arrival
        if not self.end_time or self.current_time <= self.end_time:
            self._schedule_event(self.current_time, "ARRIVAL")
        # Schedule first snapshot
        first_snapshot = self.current_time + timedelta(minutes=5)
        if not self.end_time or first_snapshot <= self.end_time:
            self._schedule_event(first_snapshot, "SNAPSHOT")

    def _schedule_event(self, time: datetime, event_type: str, data: Any = None):
        heapq.heappush(self.event_queue, (time, event_type, data))

    def _is_open_time(self, time: datetime) -> bool:
        return self.open_hour <= time.hour < self.close_hour

    def _next_open_time(self, time: datetime) -> datetime:
        if self._is_open_time(time):
            return time

        next_open = time.replace(hour=self.open_hour, minute=0, second=0, microsecond=0)
        if time.hour >= self.close_hour:
            next_open += timedelta(days=1)
        return next_open

    def _schedule_next_arrival(self, from_time: datetime):
        candidate = self._next_open_time(from_time)
        hourly_gaps = self.config.get('hourly_arrival_gaps', {})
        avg_gap = hourly_gaps.get(candidate.hour, self.config.get('avg_arrival_gap_seconds', 60))
        arrival_gap = random.expovariate(1.0 / max(avg_gap, 1))
        next_arrival = candidate + timedelta(seconds=arrival_gap)

        if not self._is_open_time(next_arrival):
            next_arrival = self._next_open_time(next_arrival)

        if self.end_time and next_arrival > self.end_time:
            return
        self._schedule_event(next_arrival, "ARRIVAL")

    def _active_lanes(self):
        lanes = [lane for lane in self.lanes if lane.active]
        if not lanes and self.lanes:
            self.lanes[0].active = True
            lanes = [self.lanes[0]]
        return lanes

    def _visible_lanes(self):
        return [lane for lane in self.lanes if lane.active or lane.current_person or lane.queue]

    def _start_ready_lanes(self):
        for lane in self._active_lanes():
            if lane.is_free() and lane.queue:
                self._handle_service_start(lane)

    def _rebalance_active_queues(self):
        active_lanes = self._active_lanes()
        if len(active_lanes) <= 1:
            return

        def lane_load(lane: Lane) -> int:
            return len(lane.queue) + (1 if lane.current_person else 0)

        # Move only waiting customers, never the person already in service.
        while True:
            busiest = max(active_lanes, key=lane_load)
            emptiest = min(active_lanes, key=lane_load)

            if busiest is emptiest:
                break
            if not busiest.queue:
                break
            if lane_load(busiest) - lane_load(emptiest) <= 1:
                break

            moved_person = busiest.queue.pop()
            emptiest.queue.append(moved_person)

    def _set_active_lane_count(self, target_lanes: int) -> int:
        target_lanes = max(1, min(5, int(target_lanes)))

        while len(self.lanes) < target_lanes:
            self.lanes.append(Lane(lane_id=len(self.lanes), camera_id=self.db.camera_id))

        active_lanes = sorted([lane for lane in self.lanes if lane.active], key=lambda lane: lane.lane_id)

        if target_lanes > len(active_lanes):
            inactive_lanes = sorted(
                [lane for lane in self.lanes if not lane.active],
                key=lambda lane: lane.lane_id,
            )
            for lane in inactive_lanes[: target_lanes - len(active_lanes)]:
                lane.active = True
            self._rebalance_active_queues()
        elif target_lanes < len(active_lanes):
            keep_ids = {lane.lane_id for lane in active_lanes[:target_lanes]}
            keep_lanes = [lane for lane in self.lanes if lane.lane_id in keep_ids and lane.active]
            for lane in active_lanes[target_lanes:]:
                lane.active = False
                while lane.queue:
                    destination = min(
                        keep_lanes,
                        key=lambda item: len(item.queue) + (1 if item.current_person else 0),
                    )
                    destination.queue.append(lane.queue.pop(0))

        self.config["num_lanes"] = target_lanes
        self._start_ready_lanes()
        return target_lanes

    def update_runtime_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        applied = {}

        if "avg_arrival_gap_seconds" in updates:
            gap = max(1, int(updates["avg_arrival_gap_seconds"]))
            self.config["avg_arrival_gap_seconds"] = gap
            applied["avg_arrival_gap_seconds"] = gap

        if "prob_caddy" in updates:
            prob_caddy = min(1.0, max(0.0, float(updates["prob_caddy"])))
            self.config["prob_caddy"] = prob_caddy
            applied["prob_caddy"] = prob_caddy

        if "prob_bakery" in updates:
            prob_bakery = min(1.0, max(0.0, float(updates["prob_bakery"])))
            self.config["prob_bakery"] = prob_bakery
            applied["prob_bakery"] = prob_bakery

        if "num_lanes" in updates:
            applied["num_lanes"] = self._set_active_lane_count(int(updates["num_lanes"]))

        return applied

    def run(self):
        print(f"Starting simulation from {self.current_time} to {self.end_time}")
        while self.event_queue:
            time, event_type, data = heapq.heappop(self.event_queue)
            if self.end_time and time > self.end_time:
                break
            
            self.current_time = time
            self._handle_event(event_type, data)

    def has_pending_events(self) -> bool:
        return bool(self.event_queue)

    def advance_until(self, target_time: datetime):
        """Advance simulation to target_time while processing all due events."""
        if self.end_time:
            target_time = min(target_time, self.end_time)

        while self.event_queue and self.event_queue[0][0] <= target_time:
            time, event_type, data = heapq.heappop(self.event_queue)
            if self.end_time and time > self.end_time:
                break
            self.current_time = time
            self._handle_event(event_type, data)

        self.current_time = target_time

    def _handle_event(self, event_type: str, data: Any):
        if event_type == "ARRIVAL":
            self._handle_arrival()
        elif event_type == "ENTRANCE_DONE":
            self._handle_entrance_done(data)
        elif event_type == "SHOPPING_DONE":
            self._handle_shopping_done(data)
        elif event_type == "SERVICE_START":
            self._handle_service_start(data)
        elif event_type == "SERVICE_END":
            self._handle_service_end(data)
        elif event_type == "SNAPSHOT":
            self._handle_snapshot()

    def _handle_arrival(self):
        if not self._is_open_time(self.current_time):
            self._schedule_next_arrival(self.current_time)
            return

        # Generate person characteristics
        gender       = random.choice(["male", "female"])
        age          = random.uniform(18, 80)
        has_bag      = random.random() < self.config.get('prob_bag',    0.30)
        has_caddy    = random.random() < self.config.get('prob_caddy',  0.20)
        is_group     = random.random() < self.config.get('prob_group',  0.10)
        picks_bakery = random.random() < self.config.get('prob_bakery', 0.25)
        group_id     = f"G{self.person_id_counter}" if is_group else None

        # Express flag resolved here so it can be stored on the person
        is_express = not has_caddy and random.random() < 0.25

        person = Person(
            track_id=self.person_id_counter,
            entry_time=self.current_time,
            gender=gender, age=age,
            has_bag=has_bag, has_caddy=has_caddy,
            is_group=is_group, group_id=group_id,
            picks_bakery=picks_bakery,
            is_express=is_express,
        )
        self.person_id_counter += 1
        self.db.insert_entrance(person)

        # ── Entrance dwell: everyone pauses at the door area ─────────────────
        # caddy: fetch trolley from rack (+30-90s)
        # bakery: browse & pick fresh items (+20-90s)
        # group: coordinate (+10-25s)
        entrance_dwell = (
            random.uniform(5, 15)
            + (random.uniform(30, 90) if has_caddy    else 0)
            + (random.uniform(20, 90) if picks_bakery else 0)
            + (random.uniform(10, 25) if is_group     else 0)
        )
        entrance_eta = self.current_time + timedelta(seconds=entrance_dwell)
        person.entrance_eta = entrance_eta
        self.people_at_entrance.append(person)
        self._schedule_event(entrance_eta, "ENTRANCE_DONE", person)

    def _handle_entrance_done(self, person: Person):
        person.entrance_done_time  = self.current_time
        person.entrance_dwell_sec  = (self.current_time - person.entry_time).total_seconds()
        self.people_at_entrance    = [p for p in self.people_at_entrance
                                      if p.track_id != person.track_id]

        # ── Schedule shop dwell ──────────────────────────────────────────────
        hour = self.current_time.hour

        # Base shop time — time-of-day aware
        if person.is_express:
            base_shop = random.uniform(2, 5)
        elif 12 <= hour <= 14:          # lunchtime — quick targeted trips
            base_shop = random.uniform(4, 10)
        elif 8 <= hour <= 10:           # morning — fuller, unhurried shops
            base_shop = random.uniform(8, 18)
        elif 17 <= hour <= 19:          # after-work — full weekly shop
            base_shop = random.uniform(10, 20)
        else:
            base_shop = random.uniform(5, 15)

        # Characteristic modifiers
        caddy_extra  = random.uniform(12, 28) if person.has_caddy else 0
        # bag without caddy → small basket, express trip
        bag_extra    = random.uniform(-6, -2) if person.has_bag and not person.has_caddy else 0
        group_extra  = random.uniform(5,  15) if person.is_group else 0
        # group + caddy = family weekly shop, significantly longer
        group_caddy  = random.uniform(5,  10) if person.is_group and person.has_caddy else 0
        # bakery picker is already browsing → slightly fuller shop
        bakery_extra = random.uniform(2,   6) if person.picks_bakery else 0
        age_extra    = (
            random.uniform(5,  12) if person.age > 65
            else random.uniform(-4,  0) if person.age < 25
            else random.uniform(0,   5) if 35 <= person.age <= 55  # prime full-shop age
            else 0
        )

        shop_time = max(2, base_shop + caddy_extra + bag_extra + group_extra
                           + group_caddy + bakery_extra + age_extra)
        person.shopping_end_eta = self.current_time + timedelta(minutes=shop_time)
        self.people_in_shop.append(person)
        self._schedule_event(person.shopping_end_eta, "SHOPPING_DONE", person)
        
        # Schedule next arrival — only during configured open hours.
        self._schedule_next_arrival(self.current_time)

    def _handle_shopping_done(self, person: Person):
        person.shopping_done_time = self.current_time
        self.people_in_shop = [p for p in self.people_in_shop if p.track_id != person.track_id]
        # Join shortest queue
        best_lane = min(
            self._active_lanes(),
            key=lambda l: len(l.queue) + (1 if l.current_person else 0),
        )
        best_lane.queue.append(person)
        
        if best_lane.is_free():
            self._schedule_event(self.current_time, "SERVICE_START", best_lane)

    def _handle_service_start(self, lane: Lane):
        if not lane.active and not lane.current_person:
            return
        if not lane.queue: return
        
        person = lane.queue.pop(0)
        lane.current_person = person
        person.service_start_time = self.current_time
        person.lane_id = lane.lane_id
        
        # Calculate service duration
        base_time = self.config.get('base_service_seconds', 30)
        bag_extra = 20 if person.has_bag else 0
        caddy_extra = 60 if person.has_caddy else 0
        group_extra = 30 if person.is_group else 0
        random_noise = random.uniform(0, 30)
        
        service_duration = base_time + bag_extra + caddy_extra + group_extra + random_noise
        lane.service_end_eta = self.current_time + timedelta(seconds=service_duration)
        self._schedule_event(lane.service_end_eta, "SERVICE_END", lane)

    def _handle_service_end(self, lane: Lane):
        person = lane.current_person
        person.service_end_time = self.current_time
        lane.current_person = None
        lane.service_end_eta = None
        
        self.db.update_dwell(person)
        self.db.log_service_event(person)
        self.completed_people.append(person)
        
        # Start next person in queue
        if lane.queue and lane.active:
            self._schedule_event(self.current_time, "SERVICE_START", lane)

    def _handle_snapshot(self):
        # Calculate queue count and avg dwell
        total_queue_count = sum(len(l.queue) + (1 if l.current_person else 0) for l in self.lanes)
        
        active_people = []
        for l in self.lanes:
            if l.current_person: active_people.append(l.current_person)
            active_people.extend(l.queue)
        
        avg_dwell = 0
        max_dwell = 0
        if active_people:
            dwell_values = [(self.current_time - p.entry_time).total_seconds() for p in active_people]
            avg_dwell = sum(dwell_values) / len(dwell_values)
            max_dwell = max(dwell_values)
        
        self.db.log_snapshot(
            self.current_time,
            total_queue_count,
            avg_dwell,
            max_dwell,
            sum(1 for lane in self.lanes if lane.active),
        )
        
        # Schedule next snapshot
        self._schedule_event(
            self.current_time + timedelta(seconds=self.snapshot_interval_seconds),
            "SNAPSHOT"
        )
