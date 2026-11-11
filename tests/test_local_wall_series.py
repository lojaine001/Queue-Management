"""
Regression tests for the DB-timestamp-to-local helpers across a DST change.

The DB session time zone is STORE_TZ, so a 30-day window that spans the
October clock change returns timestamps with two UTC offsets (+02:00 and
+01:00). pandas refuses to parse those together, which stopped every forecast
run after 2026-10-25.
"""
import ast
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from prediction.dwell_models import _to_local_series as dwell_to_local
from prediction.pipeline import to_local as pipeline_to_local

QM2 = Path(__file__).resolve().parents[1] / 'Queue-Management-System-v2-main/Queue-Management-System-v2-main'


def load_function(path, name, namespace):
    """Compile one function from a module that cannot be imported in tests (it needs the DB, models or Streamlit)."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def load_ensemble_function():
    return load_function(QM2 / 'ensemble_predict.py', '_to_local_wall_series',
                         {'pd': pd, 'datetime': datetime, 'timezone': timezone, 'STORE_TZ': 'Europe/Paris'})


class LocalWallSeriesTests(unittest.TestCase):
    def setUp(self):
        self.convert = load_ensemble_function()
        # psycopg2 returns timestamptz with a fixed-offset tzinfo per row, taken
        # from the session time zone (Europe/Paris).
        self.summer = timezone(timedelta(hours=2))
        self.winter = timezone(timedelta(hours=1))

    def test_rows_across_dst_change_keep_their_wall_clock_time(self):
        rows = [datetime(2026, 10, 24, 18, 0, tzinfo=self.summer),
                datetime(2026, 10, 26, 18, 0, tzinfo=self.winter)]
        out = self.convert(pd.Series(rows, dtype=object))
        expected = [pd.Timestamp('2026-10-24 18:00'), pd.Timestamp('2026-10-26 18:00')]
        self.assertEqual(list(out), expected)

    def test_single_offset_unchanged(self):
        rows = [datetime(2026, 11, 2, 9, 0, tzinfo=self.winter), datetime(2026, 11, 2, 9, 15, tzinfo=self.winter)]
        out = self.convert(pd.Series(rows, dtype=object))
        self.assertEqual(list(out), [pd.Timestamp('2026-11-02 09:00'), pd.Timestamp('2026-11-02 09:15')])

    def test_naive_timestamps_are_read_as_utc(self):
        out = self.convert(pd.Series([datetime(2026, 11, 2, 8, 0), datetime(2026, 7, 2, 7, 0)]))
        self.assertEqual(list(out), [pd.Timestamp('2026-11-02 09:00'), pd.Timestamp('2026-07-02 09:00')])

    def test_pipeline_to_local_across_dst_change(self):
        rows = [datetime(2026, 10, 24, 18, 0, tzinfo=self.summer), datetime(2026, 10, 26, 18, 0, tzinfo=self.winter)]
        out = pipeline_to_local(pd.Series(rows, dtype=object))
        self.assertEqual(list(out), [pd.Timestamp('2026-10-24 18:00'), pd.Timestamp('2026-10-26 18:00')])


class LocalSeriesTests(unittest.TestCase):
    """Helpers that convert to the store's (or machine's) local time zone and drop it."""

    def setUp(self):
        self.rows = pd.Series([datetime(2026, 10, 24, 18, 0, tzinfo=timezone(timedelta(hours=2))),
                               datetime(2026, 10, 26, 18, 0, tzinfo=timezone(timedelta(hours=1)))], dtype=object)
        self.expected = [pd.Timestamp('2026-10-24 18:00'), pd.Timestamp('2026-10-26 18:00')]

    def test_dwell_models_across_dst_change(self):
        os.environ.setdefault('STORE_TZ', 'Europe/Paris')
        self.assertEqual(list(dwell_to_local(self.rows)), self.expected)

    def test_dashboard_across_dst_change(self):
        convert = load_function(QM2 / 'dashboard.py', '_to_local_series',
                                {'pd': pd, 'LOCAL_TZ': ZoneInfo('Europe/Paris')})
        self.assertEqual(list(convert(self.rows)), self.expected)

    def test_naive_rows_unchanged(self):
        naive = pd.Series([datetime(2026, 11, 2, 9, 0)])
        self.assertEqual(dwell_to_local(naive).iloc[0], pd.Timestamp('2026-11-02 09:00'))

    def test_wait_forecast_conversion_across_dst_change(self):
        # Same expression as prediction/wait_forecast.py (inline there, so repeated here).
        out = pd.to_datetime(self.rows, utc=True).dt.tz_convert('Europe/Paris').dt.tz_localize(None)
        self.assertEqual(list(out), self.expected)


if __name__ == '__main__':
    unittest.main()
