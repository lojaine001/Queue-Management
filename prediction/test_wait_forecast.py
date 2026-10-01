from datetime import datetime,timedelta,timezone
import sys
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from prediction.wait_forecast import prepare_arrivals,simulate_lanes
from prediction.shared_forecast_store import DEFAULT_SETTINGS,validate_settings,publish
from prediction.forecast_state import freshness


class Cursor:
    def __init__(self,revision=1,lanes=3):self.revision=revision;self.lanes=lanes;self.sql='';self.insert=None
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def execute(self,sql,args=None):
        self.sql=sql
        if args is not None:assert sql.count('%s')==len(args)
        if 'INSERT INTO dashboard_state' in sql:self.insert=args
    def fetchone(self):return (self.revision,) if 'revision' in self.sql else (self.lanes,)
class Connection:
    def __init__(self,**kwargs):self.cur=Cursor(**kwargs)
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def cursor(self):return self.cur

class WaitWorkerTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime.now(timezone.utc)
        self.frame=pd.DataFrame(dict(ds=[self.now+timedelta(minutes=3*i+1) for i in range(8)],arrivals=[6.0]*8))

    def test_worker_import_has_no_streamlit_dependency(self):
        self.assertNotIn('streamlit',sys.modules)

    def test_lane_simulation_uses_each_candidates_backlog(self):
        payload=simulate_lanes(self.frame,4,1.,[],self.now)
        self.assertGreater(payload['lanes']['1']['slots'][0]['wait_min'],payload['lanes']['4']['slots'][0]['wait_min'])
        self.assertEqual(payload['lanes']['4']['slots'][0]['wait_min'],0)

    def test_dashboard_cannot_publish_or_recompute_shared_waits(self):
        path=Path(__file__).resolve().parents[1]/'Queue-Management-System-v2-main/Queue-Management-System-v2-main/dashboard.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                sql=node.value.upper()
                self.assertNotIn('INSERT INTO DASHBOARD_STATE',sql)
                if 'UPDATE DASHBOARD_STATE' in sql:
                    self.fail('Dashboard must not update shared state')
                    self.assertNotIn('FORECAST_JSON',sql)
                    self.assertNotIn('UPDATED_AT',sql)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                self.assertNotIn(node.func.id,('forecast_dwell','_predict_dashboard_dwell_lstm'))

    def test_latest_staffing_is_selected_at_publication(self):
        payload=simulate_lanes(self.frame,4,1.,[],self.now)
        payload.update(settings_revision=1,queue_now=4,service_min=1.)
        conn=Connection(lanes=2);publish(conn,payload)
        self.assertEqual(conn.cur.insert[2],2)
        self.assertEqual(conn.cur.insert[5],payload['lanes']['2']['wait_10m'])

    def test_obsolete_settings_cannot_publish(self):
        conn=Connection(revision=2)
        with self.assertRaisesRegex(RuntimeError,'settings changed'):
            publish(conn,{'settings_revision':1})
        self.assertIsNone(conn.cur.insert)

    def test_fresh_recomputation_does_not_hide_stale_camera_or_arrivals(self):
        state={'forecast_json':dict(as_of=self.now.isoformat(),source_published_at=(self.now-timedelta(minutes=20)).isoformat(),snapshot_at=self.now.isoformat())}
        result=freshness(state,now=self.now)
        self.assertTrue(result['stale']);self.assertEqual(result['age_seconds'],0)
        self.assertEqual(result['source_age_seconds']['source_published_at'],1200)

    def test_empty_horizon_is_unknown(self):
        payload=simulate_lanes(self.frame.iloc[:0],0,1.,[],self.now)
        self.assertIsNone(payload['lanes']['1']['wait_10m'])

    def test_arrival_selection_weighting_and_hour_calibration(self):
        frame=pd.DataFrame(dict(ds=[datetime(2026,9,27,10,0)],prophet_yhat=[10.],lstm_yhat=[20.],xgb_yhat=[30.]))
        config={**DEFAULT_SETTINGS,'arrival_models':['prophet','lstm'],'pred_smooth_min':3,'calibration_enabled':True}
        with patch.dict('os.environ',{'W_PROPHET':'.4','W_LSTM':'.3'}):
            from prediction.test_arrival_calibration import artifact
            cal=artifact(weights={'prophet':round(.4/.7,12),'lstm':round(.3/.7,12)})
            cal['k_by_hour']={'8':2.}
            result,calibrated,capped=prepare_arrivals(frame,config,cal,1000,'v1')
        self.assertAlmostEqual(result['arrivals'].iloc[0],(10*.4+20*.3)/.7*2)
        self.assertTrue(calibrated);self.assertFalse(capped)

    def test_invalid_shared_settings_rejected(self):
        for patch_value in [{'arrival_models':[]},{'pred_smooth_min':0},{'dwell_model_mode':'bad'},{'calibration_enabled':'true'}]:
            with self.assertRaises(ValueError):validate_settings({**DEFAULT_SETTINGS,**patch_value})

if __name__=='__main__':unittest.main()
