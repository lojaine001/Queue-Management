import importlib.util
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from unittest.mock import patch
from prediction.forecast_state import build_payload, wait_at_horizon

PUSH_PATH=Path(__file__).resolve().parents[1]/'Queue-Management-System-v2-main/Queue-Management-System-v2-main/push_alert_checker.py'
API_PATH=PUSH_PATH.with_name('api.py')


def load_checker():
    stubs={}
    if importlib.util.find_spec('pywebpush') is None:
        stub=types.ModuleType('pywebpush');stub.webpush=lambda **k:None
        stub.WebPushException=type('WebPushException',(Exception,),{})
        stubs['pywebpush']=stub
    with patch.dict(sys.modules,stubs):
        spec=importlib.util.spec_from_file_location('push_alert_checker_test',PUSH_PATH)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class PushAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime.now(timezone.utc)
        rows={n:[dict(ds=self.now+timedelta(minutes=3*i+1),wait_min=max(0,(4-n)*5+i-4),arrivals=5.0) for i in range(14)] for n in range(1,5)}
        self.state={'open_lanes':2,'queue_now':7,'updated_at':self.now,
                    'forecast_json':build_payload(rows,[8]*14,3,'Europe/Paris',self.now)}
        self.checker=load_checker()

    def test_push_wait_equals_forecast_wait_for_every_horizon(self):
        for minutes in (0,5,10,15,20,30,60):
            expected=wait_at_horizon(self.state,float(minutes))['wait_min']
            self.assertEqual(self.checker._wait_for_horizon(self.state,minutes),expected)

    def test_push_wait_equals_phone_endpoint(self):
        spec=importlib.util.spec_from_file_location('forecast_api_push_test',API_PATH)
        api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
        with patch.object(api,'_forecast_state',return_value=self.state):
            for minutes in (0,5,10,15):
                self.assertEqual(self.checker._wait_for_horizon(self.state,minutes),api.forecast_wait(minutes)['wait_min'])

    def test_alert_decision_uses_same_value_as_status(self):
        # an alert at threshold 5 fires exactly when the published wait is ALERT-level
        from prediction.forecast_state import status_for
        for minutes in (0,5,10,15):
            wait=self.checker._wait_for_horizon(self.state,minutes)
            self.assertEqual(wait is not None and wait>=5.0,status_for(wait)=='ALERT')

    def test_checker_does_not_read_legacy_predictions(self):
        source=PUSH_PATH.read_text(encoding='utf-8')
        self.assertNotIn('queue_predictions',source)
        self.assertNotIn('est_wait_minutes',source)

    def test_unavailable_horizon_is_none_not_zero(self):
        self.assertIsNone(self.checker._wait_for_horizon(self.state,1000))


if __name__=='__main__':unittest.main()
