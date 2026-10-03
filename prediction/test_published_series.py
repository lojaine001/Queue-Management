import unittest
from datetime import datetime, timedelta, timezone
from prediction.forecast_state import (WAIT_ALERT_MIN, WAIT_BUSY_MIN, build_payload, published_series,
                                       status_for, _HISTORY_SQL)


class Cursor:
    def __init__(self,state,history):self.state=state;self.history=history;self.sql=None;self.params=None
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def execute(self,sql,args=None):
        self.sql=sql
        if 'wait_forecast_runs' in sql:self.history_params=args
    def fetchone(self):return (self.state['forecast_json'],self.state['open_lanes'])
    def fetchall(self):return list(self.history)


class Connection:
    def __init__(self,state,history=()):self.cur=Cursor(state,history)
    def cursor(self,**kwargs):return self.cur


class PublishedSeriesTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,10,1,12,0,tzinfo=timezone.utc)
        rows={n:[dict(ds=self.now+timedelta(minutes=3*i+1),wait_min=(5-n)*2.0+i,arrivals=4.0+i) for i in range(6)] for n in (1,2,3)}
        self.state=dict(forecast_json=build_payload(rows,[8]*6,3,'Europe/Paris',self.now),open_lanes=2)

    def test_status_thresholds_are_shared(self):
        self.assertEqual((WAIT_BUSY_MIN,WAIT_ALERT_MIN),(2.0,5.0))
        self.assertEqual([status_for(v) for v in (None,float('nan'),0,1.99,2.0,4.99,5.0,26)],
                         [None,None,'OK','OK','BUSY','BUSY','ALERT','ALERT'])

    def test_current_slots_use_selected_lanes_and_carry_status(self):
        rows=published_series(Connection(self.state),self.now,self.now+timedelta(hours=1))
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[0]['wait_min'],6.0)  # lane 2 scenario
        self.assertEqual(rows[0]['arrivals'],8.0)
        self.assertEqual([r['status'] for r in rows],[status_for(r['wait_min']) for r in rows])

    def test_history_is_cut_at_first_current_slot_and_ordered_before_current(self):
        past=[(self.now-timedelta(minutes=3*i),3.0,7.0) for i in (3,2,1)]
        conn=Connection(self.state,past)
        rows=published_series(conn,self.now-timedelta(hours=1),self.now+timedelta(hours=1))
        self.assertEqual(len(rows),9)
        self.assertEqual([r['wait_min'] for r in rows[:3]],[7.0]*3)
        self.assertEqual(rows[0]['status'],'ALERT')
        self.assertEqual(conn.cur.history_params['cut'],self.now+timedelta(minutes=1))
        self.assertIn('prediction_for < %(cut)s',_HISTORY_SQL)

    def test_history_uses_last_publication_per_slot_and_its_selected_lanes(self):
        sql=_HISTORY_SQL
        self.assertIn('DISTINCT ON (prediction_for)',sql)
        self.assertIn('ORDER BY prediction_for, published_at DESC',sql)
        self.assertIn("r.payload->'lanes'->(r.selected_lanes::text)",sql)

    def test_current_slots_outside_range_are_excluded(self):
        rows=published_series(Connection(self.state),self.now+timedelta(minutes=7),self.now+timedelta(minutes=13))
        self.assertEqual(len(rows),2)
        self.assertEqual(published_series(Connection(self.state),self.now-timedelta(days=2),self.now-timedelta(days=1)),[])

    def test_no_data_before_history_began_and_no_fallback(self):
        state=dict(forecast_json=None,open_lanes=2)
        conn=Connection(state,[])
        self.assertEqual(published_series(conn,datetime(2026,9,1,tzinfo=timezone.utc),datetime(2026,9,2,tzinfo=timezone.utc)),[])
        self.assertNotIn('queue_predictions',conn.cur.sql)

    def test_naive_bounds_are_store_time_converted_to_utc(self):
        conn=Connection(self.state,[])
        published_series(conn,datetime(2026,10,1,14,0),datetime(2026,10,1,15,0))
        self.assertEqual(conn.cur.history_params['start'],datetime(2026,10,1,12,0,tzinfo=timezone.utc))


if __name__=='__main__':unittest.main()
