import unittest
from prediction.schema_check import missing_tables, require_tables


class Cursor:
    def __init__(self, missing, as_dict=False):
        self.missing, self.as_dict, self.calls = missing, as_dict, []
    def execute(self, sql, args=None): self.calls.append((sql, args))
    def fetchall(self): return [{"t": t} if self.as_dict else (t,) for t in self.missing]


class SchemaCheckTests(unittest.TestCase):
    def test_all_tables_present(self):
        cur = Cursor([])
        require_tables(cur, "forecast_settings", "dashboard_state")
        sql, args = cur.calls[0]
        self.assertIn("to_regclass", sql)
        self.assertEqual(args, (["forecast_settings", "dashboard_state"],))

    def test_missing_table_stops_with_the_command_to_run(self):
        with self.assertRaises(RuntimeError) as ctx:
            require_tables(Cursor(["wait_forecast_runs"]), "forecast_settings", "wait_forecast_runs")
        self.assertIn("wait_forecast_runs", str(ctx.exception))
        self.assertIn("alembic -c db/alembic.ini upgrade head", str(ctx.exception))

    def test_dict_cursor(self):
        self.assertEqual(missing_tables(Cursor(["lane_events"], as_dict=True), ["lane_events"]), ["lane_events"])


if __name__ == "__main__":
    unittest.main()
