"""Compatibility entrypoint for the production dashboard.

This file stays small on purpose so the canonical implementation lives in one place.
"""

from pathlib import Path
import runpy


CANONICAL_DASHBOARD = Path(__file__).resolve().parent / "Queue-Management-System-v2-main" / "dashboard.py"
runpy.run_path(str(CANONICAL_DASHBOARD), run_name="__main__")
