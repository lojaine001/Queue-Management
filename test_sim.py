import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from simulator.engine import SimulatorEngine
from simulator.scenarios import get_scenario
from unittest.mock import MagicMock

class DummyDB:
    camera_id = "test_cam"
    def insert_entrance(self, p): pass
    def update_dwell(self, p): pass
    def log_service_event(self, p): pass
    def log_snapshot(self, t, q, a, m, l):
        print(f"[{t}] Snapshot: Queue={q}, AvgDwell={a}, Lanes={l}")
    def close(self): pass

scenario = get_scenario("normal_day")
engine = SimulatorEngine(scenario, DummyDB())
engine.run()
print("Processed people:", len(engine.completed_people))
print("People spawned:", engine.person_id_counter)
