"""
Reproduction script for missing optional package import failure in dashboard.py.
Simulates importing dashboard.py in an environment where optional ML dependencies (e.g. xgboost) are absent.
"""
import sys
import unittest.mock as mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QM = ROOT / "Queue-Management"
APP = QM / "Queue-Management-System-v2-main" / "Queue-Management-System-v2-main"

sys.path = [str(QM), str(APP), str(ROOT)] + [p for p in sys.path if p not in (str(QM), str(APP), str(ROOT))]


def test_import_with_masked_module(module_name):
    print(f"[*] Simulating import of dashboard.py with '{module_name}' missing...")
    # Hide module from sys.modules
    original_modules = sys.modules.copy()
    sys.modules[module_name] = None  # In Python 3.3+, setting to None raises ModuleNotFoundError

    try:
        # Attempt to import or exec dashboard module
        import importlib
        if "dashboard" in sys.modules:
            del sys.modules["dashboard"]

        dashboard_file = APP / "dashboard.py"
        with open(dashboard_file, "r", encoding="utf-8") as f:
            code = f.read()

        # Compile and execute the top-level statements
        bytecode = compile(code, str(dashboard_file), "exec")
        exec_globals = {"__file__": str(dashboard_file), "__name__": "dashboard_repro"}
        exec(bytecode, exec_globals)
        print(f"  => SUCCESS: dashboard loaded gracefully without '{module_name}'")
        return False
    except (ModuleNotFoundError, ImportError) as exc:
        import traceback
        traceback.print_exc()
        print(f"  => REPRODUCED {type(exc).__name__}: {exc}")
        return True
    finally:
        # Restore sys.modules
        sys.modules.clear()
        sys.modules.update(original_modules)


def main():
    reproduced = test_import_with_masked_module("xgboost")
    print("\n[Result Summary]")
    if reproduced:
        print("  => REPRODUCTION SUCCESSFUL: Unprotected top-level import crashes dashboard when dependency is missing.")
    else:
        print("  => Dependency failure was handled gracefully.")


if __name__ == "__main__":
    main()
