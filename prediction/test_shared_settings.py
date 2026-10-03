import unittest
from prediction.shared_forecast_store import DEFAULT_SETTINGS, ensure_schema, validate_settings


class Cursor:
    def __init__(self):self.calls=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def execute(self,sql,args=None):self.calls.append((sql,args))


class Connection:
    def __init__(self):self.cur=Cursor();self.committed=False
    def cursor(self,**kwargs):return self.cur
    def commit(self):self.committed=True


class SharedSettingsTests(unittest.TestCase):
    def test_default_coefficient_is_one(self):
        self.assertEqual(DEFAULT_SETTINGS['arrival_coefficient'],1.0)
        validate_settings(dict(DEFAULT_SETTINGS))

    def test_coefficient_range_and_type_enforced(self):
        for bad in [0.49,2.01,0,-1,float('nan'),float('inf'),'1.0',None,True]:
            with self.assertRaises(ValueError,msg=repr(bad)):
                validate_settings({**DEFAULT_SETTINGS,'arrival_coefficient':bad})
        for good in [0.5,1,1.25,2.0]:
            validate_settings({**DEFAULT_SETTINGS,'arrival_coefficient':good})

    def test_missing_key_is_rejected_not_defaulted(self):
        old={k:v for k,v in DEFAULT_SETTINGS.items() if k!='arrival_coefficient'}
        with self.assertRaises(ValueError):validate_settings(old)

    def test_schema_backfills_old_rows_and_existing_values_win(self):
        conn=Connection();ensure_schema(conn)
        sql,args=next((s,a) for s,a in conn.cur.calls if s.startswith('UPDATE forecast_settings SET config'))
        # `defaults || stored`: stored values override defaults, so only missing keys are filled.
        self.assertIn('%s::jsonb || config',sql)
        self.assertIn('NOT config ?& %s',sql)
        self.assertIn('arrival_coefficient',args[1])
        self.assertEqual(set(args[1]),set(DEFAULT_SETTINGS))
        old={'arrival_models':['prophet'],'pred_smooth_min':15}
        merged={**DEFAULT_SETTINGS,**old}
        self.assertEqual(merged['arrival_coefficient'],1.0)
        self.assertEqual(merged['pred_smooth_min'],15)
        self.assertTrue(conn.committed)


if __name__=='__main__':unittest.main()
