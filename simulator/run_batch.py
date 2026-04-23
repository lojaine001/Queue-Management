import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.engine import SimulatorEngine
from simulator.db import SimDB
from simulator.scenarios import get_scenario

def run_sim(scenario_name: str):
    config = get_scenario(scenario_name)
    db = SimDB(scenario_name)
    
    engine = SimulatorEngine(config, db)
    engine.run()
    
    print(f"Simulation {scenario_name} complete.")
    print(f"Total people processed: {len(engine.completed_people)}")
    db.close()

if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "normal_day"
    run_sim(scenario)
