from datetime import datetime,timezone,timedelta
import unittest
import pandas as pd
from prediction.arrival_calibration import calibration_status,factors,target_definition
from prediction.accuracy import evaluate


def artifact(**changes):
    result=dict(schema=2,artifact_id='test',target=target_definition(),model_version='v1',weights={'prophet':1.},
                timezone='UTC',k_global=1.5,k_by_hour={},expires_at=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),
                validation=dict(passed=True,fit_days=7,holdout_days=7,raw_mae=5,calibrated_mae=0))
    result.update(changes);return result


class CalibrationTests(unittest.TestCase):
    def test_timezone_is_dst_aware(self):
        a=artifact(k_by_hour={'8':2.,'9':3.})
        self.assertEqual(factors([datetime(2026,9,27,10),datetime(2026,12,27,10)],a),[2.,3.])
    def test_legacy_malformed_and_incompatible_artifacts_are_not_applied(self):
        for a in [[],{},dict(k_global=1.348),artifact(k_global=float('nan')),artifact(k_global=-1),artifact(model_version='v0'),artifact(weights={'lstm':1.}),artifact(timezone='bad'),artifact(expires_at='2020-01-01T00:00:00+00:00')]:
            self.assertFalse(calibration_status(a,True,'v1',{'prophet':1.})['applied'])
    def test_disabled_never_applies(self):
        self.assertFalse(calibration_status(artifact(),False,'v1',{'prophet':1.})['applied'])
    def test_valid_artifact_applies(self):
        self.assertTrue(calibration_status(artifact(),True,'v1',{'prophet':1.})['applied'])
    def frame(self,second_half_actual=15.):
        dates=pd.date_range('2026-08-01',periods=14,freq='D',tz='UTC')
        return pd.DataFrame([dict(ds=d+timedelta(minutes=3*i),model_version='v1',prophet=10.,actual=15. if j<7 else second_half_actual) for j,d in enumerate(dates) for i in range(20)])
    def test_fit_raw_and_score_later_holdout(self):
        report,a=evaluate(self.frame(),{'prophet':1.},'v1')
        self.assertEqual(a['k_global'],1.5);self.assertEqual(report['holdout_calibrated']['mae'],0)
        self.assertLess(a['fit_end'],a['holdout_start'])
    def test_future_reversal_fails_even_when_fit_improves(self):
        report,a=evaluate(self.frame(10.),{'prophet':1.},'v1')
        self.assertIsNone(a);self.assertEqual(report['status'],'holdout_failed')
    def test_short_history_cannot_make_candidate(self):
        report,a=evaluate(self.frame().iloc[:20],{'prophet':1.},'v1')
        self.assertIsNone(a);self.assertEqual(report['status'],'insufficient_data')
    def test_zeros_are_scored(self):
        report,a=evaluate(self.frame(0.),{'prophet':1.},'v1')
        self.assertEqual(report['zero_buckets'],140)
        self.assertEqual(report['holdout_raw']['mae'],10.)

if __name__=='__main__':unittest.main()
