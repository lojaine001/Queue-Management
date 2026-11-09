"""
Regression test for ensemble_predict._to_local_wall_series across a DST change.

The DB session time zone is STORE_TZ, so a 30-day window that spans the
October clock change returns timestamps with two UTC offsets (+02:00 and
+01:00). pandas refuses to parse those together, which stopped every forecast
run after 2026-10-25.
"""
import ast
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ENSEMBLE = (Path(__file__).resolve().parents[1]
            / 'Queue-Management-System-v2-main/Queue-Management-System-v2-main/ensemble_predict.py')


def load_function():
    """Compile only _to_local_wall_series from ensemble_predict.py (the module itself needs the DB and models)."""
    tree = ast.parse(ENSEMBLE.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_to_local_wall_series')
    namespace = {'pd': pd, 'datetime': datetime, 'timezone': timezone, 'STORE_TZ': 'Europe/Paris'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ENSEMBLE), 'exec'), namespace)
    return namespace['_to_local_wall_series']


class LocalWallSeriesTests(unittest.TestCase):
    def setUp(self):
        self.convert = load_function()
        self.offset = datetime.now(timezone.utc).astimezone().utcoffset()
        # psycopg2 returns timestamptz with a fixed-offset tzinfo per row, taken
        # from the session time zone (Europe/Paris).
        self.summer = timezone(timedelta(hours=2))
        self.winter = timezone(timedelta(hours=1))

    def test_rows_across_dst_change_keep_their_wall_clock_time(self):
        rows = [datetime(2026, 10, 24, 18, 0, tzinfo=self.summer),
                datetime(2026, 10, 26, 18, 0, tzinfo=self.winter)]
        out = self.convert(pd.Series(rows, dtype=object))
        expected = [pd.Timestamp('2026-10-24 18:00') + self.offset, pd.Timestamp('2026-10-26 18:00') + self.offset]
        self.assertEqual(list(out), expected)

    def test_single_offset_unchanged(self):
        rows = [datetime(2026, 11, 2, 9, 0, tzinfo=self.winter), datetime(2026, 11, 2, 9, 15, tzinfo=self.winter)]
        out = self.convert(pd.Series(rows, dtype=object))
        self.assertEqual(list(out), [pd.Timestamp('2026-11-02 09:00') + self.offset,
                                     pd.Timestamp('2026-11-02 09:15') + self.offset])

    def test_naive_timestamps_unchanged(self):
        out = self.convert(pd.Series([datetime(2026, 11, 2, 9, 0)]))
        self.assertEqual(out.iloc[0], pd.Timestamp('2026-11-02 09:00') + self.offset)


if __name__ == '__main__':
    unittest.main()
