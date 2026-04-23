import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.engine import SimulatorEngine
from simulator.db import SimDB
from simulator.scenarios import get_scenario

def test_run():
    scenario_name = "test_rush"
    config = {
        "num_lanes": 2,
        "avg_arrival_gap_seconds": 30,
        "prob_bag": 0.5,
        "prob_caddy": 0.2,
        "prob_group": 0.1,
        "base_service_seconds": 30,
        "start_time": datetime.now(),
        "end_time": datetime.now() + timedelta(hours=1)
    }
    
    db = SimDB(scenario_name)
    engine = SimulatorEngine(config, db)
    engine.run()
    
    print(f"Test simulation {scenario_name} complete.")
    print(f"Total people processed: {len(engine.completed_people)}")
    db.close()

if __name__ == "__main__":
    test_run()
