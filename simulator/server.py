import asyncio
import json
import websockets
import websockets.exceptions
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.calibrate import calibrate
from simulator.db import SimDB
from simulator.engine import SimulatorEngine
from simulator.scenarios import SCENARIOS, get_scenario

LOOPBACK_HOST = "127.0.0.1"

class NullDB:
    def __init__(self, scenario_name: str):
        self.camera_id = f"SIM_PREVIEW_{scenario_name}"

    def insert_entrance(self, person):
        pass

    def update_dwell(self, person):
        pass

    def log_snapshot(self, timestamp, queue_count, avg_dwell, max_dwell, active_lanes):
        pass

    def log_service_event(self, person):
        pass

    def close(self):
        pass

connected = set()
current_engine = None
simulation_speed = 30.0
is_running = False

BASE_SERVICE_SEC = 30   # matches engine default
AVG_NOISE_SEC    = 15   # expected value of uniform(0, 30)


def next_open_time(dt, open_hour: int, close_hour: int):
    if open_hour <= dt.hour < close_hour:
        return dt

    next_open = dt.replace(hour=open_hour, minute=0, second=0, microsecond=0)
    if dt.hour >= close_hour:
        next_open += timedelta(days=1)
    return next_open

def serialize_person(person, state, current_time):
    entry        = getattr(person, 'entry_time', None)
    queue_since  = getattr(person, 'shopping_done_time', None)
    svc_since    = getattr(person, 'service_start_time', None)

    dwell_sec       = round((current_time - entry).total_seconds())       if entry       else 0
    queue_wait_sec  = round((current_time - queue_since).total_seconds())  if queue_since  else 0
    service_wait_sec= round((current_time - svc_since).total_seconds())   if svc_since   else 0

    return {
        "id":               getattr(person, 'track_id', 0),
        "gender":           getattr(person, 'gender', 'unknown'),
        "is_group":         getattr(person, 'is_group', False),
        "has_bag":          getattr(person, 'has_bag', False),
        "has_caddy":        getattr(person, 'has_caddy', False),
        "picks_bakery":     getattr(person, 'picks_bakery', False),
        "is_express":       getattr(person, 'is_express', False),
        "entry_time":       entry.isoformat() if entry else None,
        "shopping_end_eta": (person.shopping_end_eta.isoformat()
                             if getattr(person, 'shopping_end_eta', None) else None),
        "age":              round(getattr(person, 'age', 35) or 35),
        "dwell_sec":        dwell_sec,
        "queue_wait_sec":   queue_wait_sec,
        "service_wait_sec": service_wait_sec,
        "state":            state,
    }


def serialize_entrance_person(person, current_time):
    entry = getattr(person, 'entry_time', None)
    eta   = getattr(person, 'entrance_eta', None)
    elapsed = (current_time - entry).total_seconds() if entry else 0
    total   = (eta - entry).total_seconds() if eta and entry else max(elapsed, 1)
    progress = round(min(elapsed / max(total, 1), 1.0), 3)
    data = serialize_person(person, "entrance", current_time)
    data["entrance_progress"] = progress
    return data


def _estimate_service_sec(person):
    return (BASE_SERVICE_SEC
            + (20 if person.has_bag   else 0)
            + (60 if person.has_caddy else 0)
            + (30 if person.is_group  else 0)
            + AVG_NOISE_SEC)


def _lane_wait_sec(lane, current_time):
    """Estimated seconds until the last queued person finishes service."""
    if not lane.current_person:
        return 0
    # Remaining time for the person currently being served
    if lane.service_end_eta:
        remaining = max(0, (lane.service_end_eta - current_time).total_seconds())
    else:
        remaining = BASE_SERVICE_SEC
    # Add estimated service time for each waiting person
    for p in lane.queue:
        remaining += _estimate_service_sec(p)
    return round(remaining)


async def simulation_loop():
    global current_engine, is_running

    while True:
        if current_engine and is_running and connected:
            if current_engine.has_pending_events() and current_engine.current_time < current_engine.end_time:
                dt = 1.0 / 30.0
                target_time = current_engine.current_time + timedelta(seconds=dt * simulation_speed)
                current_engine.advance_until(target_time)
                now = current_engine.current_time

                state = {
                    "type":      "update",
                    "time":      now.isoformat(),
                    "entrance":  [serialize_entrance_person(p, now) for p in current_engine.people_at_entrance],
                    "shop":      [serialize_person(p, "shopping", now) for p in current_engine.people_in_shop],
                    "lanes":     [],
                    "completed": len(current_engine.completed_people),
                }
                for lane in current_engine._visible_lanes():
                    l_data = {
                        "lane_id":  lane.lane_id,
                        "active":   lane.active,
                        "current":  (serialize_person(lane.current_person, "service", now)
                                     if lane.current_person else None),
                        "queue":    [serialize_person(p, "queue", now) for p in lane.queue],
                        "wait_sec": _lane_wait_sec(lane, now),
                    }
                    state["lanes"].append(l_data)
                
                msg = json.dumps(state)
                websockets.broadcast(connected, msg)
            elif not current_engine.has_pending_events() or current_engine.current_time >= current_engine.end_time:
                is_running = False
                msg = json.dumps({"type": "status", "message": "Simulation Complete"})
                websockets.broadcast(connected, msg)
                
        await asyncio.sleep(1.0 / 30.0)

async def handler(websocket):
    global current_engine, is_running, simulation_speed
    connected.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("action") == "start":
                scenario = data.get("scenario", "normal_day")
                config = get_scenario(scenario)
                simulation_speed = float(data.get("speed", 30))
                do_calibrate = data.get("calibrate", False)
                calibration_days = int(data.get("calibration_days", 30))
                do_persist = data.get("persist", False)
                sim_hours = max(1, min(12, int(data.get("hours", 8))))
                start_mode = str(data.get("start_mode", "Now"))
                start_datetime = data.get("start_datetime")
                
                if do_calibrate:
                    try:
                        print(f"[server] Applying calibration from last {calibration_days} days of REAL data")
                        config.update(calibrate(days=calibration_days))
                    except Exception as e:
                        print(f"Calibration error: {e}")

                # Apply Streamlit UI overrides after calibration so user choices win.
                config["num_lanes"] = int(data.get("lanes", config.get("num_lanes", 2)))
                config["avg_arrival_gap_seconds"] = int(
                    data.get("gap", config.get("avg_arrival_gap_seconds", 30))
                )
                config["prob_caddy"] = float(
                    data.get("prob_caddy", config.get("prob_caddy", 0.2))
                )
                config["prob_bakery"] = float(
                    data.get("prob_bakery", config.get("prob_bakery", 0.25))
                )

                open_hour = int(config.get("open_hour", 8))
                close_hour = int(config.get("close_hour", 20))

                if start_mode == "Specific date/time" and start_datetime:
                    try:
                        requested_start = datetime.fromisoformat(start_datetime)
                    except ValueError:
                        requested_start = datetime.now()
                else:
                    requested_start = datetime.now()

                effective_start = next_open_time(requested_start, open_hour, close_hour)
                scenario_end = effective_start.replace(
                    hour=close_hour, minute=0, second=0, microsecond=0
                )
                requested_end = effective_start + timedelta(hours=sim_hours)

                config["start_time"] = effective_start
                config["end_time"] = min(scenario_end, requested_end)
                
                db = SimDB(f"WS_{scenario}") if do_persist else NullDB(scenario)
                current_engine = SimulatorEngine(config, db)
                is_running = True

                # Broadcast to all connected clients (frontend canvas).
                # Do NOT await websocket.send() — the Streamlit command client
                # closes immediately after sending, so its socket is already gone.
                websockets.broadcast(
                    connected,
                    json.dumps({
                        "type": "status",
                        "message": "Simulation Started",
                        "effective_start": effective_start.isoformat(),
                        "effective_end": config["end_time"].isoformat(),
                    })
                )

            elif data.get("action") == "stop":
                is_running = False
                websockets.broadcast(
                    connected,
                    json.dumps({"type": "status", "message": "Simulation Stopped"})
                )
            elif data.get("action") == "update_config":
                if current_engine is None:
                    continue

                updates = {}
                if "lanes" in data:
                    updates["num_lanes"] = int(data["lanes"])
                if "gap" in data:
                    updates["avg_arrival_gap_seconds"] = int(data["gap"])
                if "prob_caddy" in data:
                    updates["prob_caddy"] = float(data["prob_caddy"])
                if "prob_bakery" in data:
                    updates["prob_bakery"] = float(data["prob_bakery"])
                if "speed" in data:
                    simulation_speed = float(data["speed"])

                applied = current_engine.update_runtime_config(updates)
                applied["speed"] = simulation_speed
                websockets.broadcast(
                    connected,
                    json.dumps({
                        "type": "status",
                        "message": "Live config updated",
                        "applied": applied,
                    })
                )
    except websockets.exceptions.ConnectionClosedError:
        pass   # normal client disconnect — not an error
    finally:
        connected.remove(websocket)

async def main():
    print(f"WebSocket server running on ws://{LOOPBACK_HOST}:8080")
    asyncio.create_task(simulation_loop())
    async with websockets.serve(handler, LOOPBACK_HOST, 8080,
                                ping_interval=None,   # no keepalive pings on loopback
                                ping_timeout=None):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
