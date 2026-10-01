import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from prediction.forecast_state import build_payload, freshness, waiting_backlog, wait_at_horizon, horizon_value

API_PATH=Path(__file__).resolve().parents[1]/'Queue-Management-System-v2-main/Queue-Management-System-v2-main/api.py'
spec=importlib.util.spec_from_file_location('forecast_api_test',API_PATH)
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)


class FakeConnection:
    def __init__(self,state):self.state=state;self.committed=False
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def cursor(self,**kwargs):return self
    def execute(self,sql,args=None):
        if args is not None:assert sql.count('%s')==len(args)
        if 'UPDATE dashboard_state' in sql:
            self.state.update(dict(zip(['open_lanes','wait_0m','wait_5m','wait_10m','wait_15m'],args)))
    def fetchone(self):return self.state
    def commit(self):self.committed=True
    def close(self):pass


class SharedForecastTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime.now(timezone.utc)
        rows={n:[dict(ds=self.now+timedelta(minutes=3*i+1),wait_min=max(0,(4-n)*5+i-4)) for i in range(8)] for n in range(1,5)}
        self.state={'open_lanes':4,'queue_now':7,'updated_at':self.now,
                    'forecast_json':build_payload(rows,[8]*8,3,'Europe/Paris',self.now)}

    def test_nonlinear_lane_scenarios_preserved_when_current_wait_is_zero(self):
        with patch.object(api,'_forecast_state',return_value=self.state):response=api.forecast()
        self.assertEqual(response['wait_10_min'],0)
        scenarios=response['lane_scenarios']
        self.assertEqual(scenarios[0]['est_wait_min'],14)
        self.assertEqual(scenarios[1]['est_wait_min'],9)
        self.assertTrue(scenarios[3]['is_current'])

    def test_fixed_and_arbitrary_horizons_agree_for_same_snapshot(self):
        with patch.object(api,'_forecast_state',return_value=self.state):
            fixed=api.forecast()
            for minutes,key in [(0,'wait_now_min'),(5,'wait_5_min'),(10,'wait_10_min'),(15,'wait_15_min')]:
                self.assertEqual(api.forecast_wait(minutes)['wait_min'],fixed[key])
            chart=api.forecast_chart()
            self.assertEqual(chart['slots'][3]['wait_min'],fixed['wait_10_min'])
            self.assertEqual(chart['current_lanes'],4)

    def test_lane_change_updates_waits_without_changing_input_age(self):
        conn=FakeConnection(self.state)
        with patch.object(api,'_conn',return_value=conn):
            api.set_lanes(api.SetLanesRequest(lanes=2))
        self.assertEqual(self.state['open_lanes'],2)
        self.assertEqual(self.state['wait_10m'],9)
        self.assertEqual(self.state['updated_at'],self.now)
        self.assertEqual(freshness(self.state)['updated_at'],self.now.isoformat())
        self.assertTrue(conn.committed)

    def test_stale_lane_change_rejected_without_mutation(self):
        self.state['forecast_json']['as_of']=(self.now-timedelta(minutes=20)).isoformat()
        conn=FakeConnection(self.state)
        with patch.object(api,'_conn',return_value=conn),self.assertRaises(HTTPException) as caught:
            api.set_lanes(api.SetLanesRequest(lanes=2))
        self.assertEqual(caught.exception.status_code,409)
        self.assertEqual(self.state['open_lanes'],4)
        self.assertFalse(conn.committed)
        self.assertTrue(freshness(self.state)['stale'])

    def test_missing_forecast_is_unknown_not_zero_or_500(self):
        with patch.object(api,'_forecast_state',return_value={'open_lanes':3,'wait_0m':None}):
            response=api.forecast()
        self.assertIsNone(response['wait_now_min'])
        self.assertEqual(response['lane_scenarios'],[])
        self.assertTrue(response['stale'])

    def test_invalid_and_unavailable_horizons(self):
        for minutes in [-1,float('nan'),float('inf')]:
            with patch.object(api,'_forecast_state',return_value=self.state),self.assertRaises(HTTPException) as caught:
                api.forecast_wait(minutes)
            self.assertEqual(caught.exception.status_code,422)
        self.assertIsNone(wait_at_horizon(self.state,1000)['wait_min'])

    def test_horizon_uses_timestamp_not_fixed_row_offset(self):
        rows=[dict(ds=self.now+timedelta(minutes=2+3*i),wait_min=10*i) for i in range(6)]
        self.assertEqual(horizon_value(rows,5,self.now),10)
        self.assertEqual(horizon_value(rows,10,self.now),30)
        self.assertIsNone(horizon_value(rows,30,self.now))

    def test_naive_store_times_are_saved_as_utc(self):
        reference=datetime(2026,1,1,11,tzinfo=timezone.utc)
        payload=build_payload({1:[dict(ds=datetime(2026,1,1,12,3),wait_min=2)]},[8],3,'Europe/Paris',reference)
        self.assertEqual(payload['lanes']['1']['slots'][0]['prediction_for'],'2026-01-01T11:03:00+00:00')

    def test_backlog_recomputed_per_lane_scenario(self):
        self.assertEqual(waiting_backlog(4,4),0)
        self.assertEqual(waiting_backlog(4,1),3)
        self.assertEqual(waiting_backlog(1,4),0)

class PendingSettingsTests(unittest.TestCase):
    def test_revision_difference_is_pending(self):
        from prediction.forecast_state import freshness
        state={'current_settings_revision':8,'forecast_json':{'settings_revision':7,'as_of':datetime.now(timezone.utc).isoformat()}}
        self.assertTrue(freshness(state)['pending_settings'])
        state['forecast_json']['settings_revision']=8
        self.assertFalse(freshness(state)['pending_settings'])

if __name__=='__main__':unittest.main()
