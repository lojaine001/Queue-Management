"""Exercise the actual publication boundary without importing/training ML models."""
import ast
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo
import pandas as pd
from prediction.runtime import write_json, read_json

APP=Path(__file__).resolve().parents[1]/'Queue-Management-System-v2-main/Queue-Management-System-v2-main/ensemble_predict.py'

class Connection:
    def __init__(self, fail=False):
        self.statements=[]; self.committed=False; self.fail=fail; self.rolled_back=False
    def cursor(self): return self
    def execute(self, sql, args=None):
        if args is not None:
            assert sql.count("%s") == len(args), "SQL placeholder count must match arguments"
        self.statements.append((sql,args))
    def commit(self):
        if self.fail: raise RuntimeError('deliberate commit failure')
        self.committed=True
    def rollback(self): self.rolled_back=True
    def close(self): pass

class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.conn=Connection()
        self.env={'datetime':datetime,'timezone':timezone,'ZoneInfo':ZoneInfo,'STORE_TZ':'Europe/Paris',
                  'pd':pd,'DB_CONFIG':{},'BUCKET_MINUTES':3,'RUNTIME':Path(self.tmp.name),
                  'write_json':write_json,'_connect':lambda:self.conn}
        tree=ast.parse(APP.read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_save_to_db','_print_results')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(APP),'exec'),self.env)
        now=datetime.now(timezone.utc)
        self.result={
            'prophet_preds':pd.DataFrame({'ds':[now-timedelta(minutes=3),now+timedelta(minutes=3)]}),
            'prophet_vals':[1.,2.],'lstm_vals':[1.,2.],'xgb_vals':[1.,2.],'ensemble_vals':[1.,2.],
            'checkout_arrivals':[1.,2.],'wait_estimates':[1.,2.], 'lane_waits_15m':{1:3,2:2,3:1},
            'source':'REAL','run_id':'test-id','model_version':'test-version','data_cutoff':now.isoformat(),
            'forecast_origin':now.isoformat(),'run_started_at':now.isoformat(),
            'service_per_bucket':3,'active_lanes':3,'browsing_gap_min':25,'wait_15m':1.,'wait_30m':2.}
    def tearDown(self): self.tmp.cleanup()
    def test_printing_results_has_no_publication_dependencies(self):
        with redirect_stdout(io.StringIO()):self.env['_print_results'](self.result)
    def test_only_future_rows_published_and_vintage_committed_together(self):
        with redirect_stdout(io.StringIO()):self.env['_save_to_db'](self.result)
        self.assertTrue(self.conn.committed)
        live=[s for s in self.conn.statements if 'INSERT INTO queue_predictions' in s[0]]
        history=[s for s in self.conn.statements if 'INSERT INTO forecast_runs' in s[0]]
        self.assertEqual(len(live),1);self.assertEqual(len(history),1)
        self.assertEqual(len(history[0][1][-1].adapted),1)
        self.assertEqual(read_json(Path(self.tmp.name)/'publication.json')['run_id'],'test-id')
    def test_failed_commit_does_not_claim_success(self):
        self.conn.fail=True
        with redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'commit failure'):
            self.env['_save_to_db'](self.result)
        self.assertFalse((Path(self.tmp.name)/'publication.json').exists())
    def test_expired_forecast_is_rejected(self):
        self.result['prophet_preds']['ds']=datetime.now(timezone.utc)-timedelta(minutes=10)
        with redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'No future'):
            self.env['_save_to_db'](self.result)
        self.assertTrue(self.conn.rolled_back)
        self.assertFalse(self.conn.committed)

if __name__=='__main__':unittest.main()
