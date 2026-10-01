import ast
import os
from pathlib import Path
import tempfile
import unittest
from prediction.model_bundle import ModelBundle, current_bundle
from prediction.runtime import read_json


class BundleTests(unittest.TestCase):
    def test_failed_training_keeps_previous_complete_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            models=Path(tmp)
            (models/'weights').write_text('old')
            (models/'scaler').write_text('old')
            bundle=ModelBundle(models, ['weights','scaler'])
            (bundle.stage/'weights').write_text('new')
            (bundle.stage/'scaler').unlink()
            with self.assertRaises(RuntimeError): bundle.promote(['weights','scaler'], {})
            self.assertEqual(current_bundle(models), models)
            self.assertEqual((models/'weights').read_text(), 'old')
            bundle.cleanup()
            self.assertFalse(bundle.stage.exists())

    def test_publication_switches_complete_generation_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            models=Path(tmp)
            (models/'weights').write_text('old')
            bundle=ModelBundle(models,['weights'])
            self.assertFalse(bundle.changed())
            (bundle.stage/'weights').write_text('replacement')
            self.assertTrue(bundle.changed())
            self.assertEqual(current_bundle(models),models)
            bundle.promote(['weights'], {'source':'REAL'})
            bundle.cleanup()
            self.assertEqual((current_bundle(models)/'weights').read_text(),'replacement')
            self.assertEqual((models/'weights').read_text(),'old')

    def test_cache_reuses_new_data_but_respects_config_and_age(self):
        path=Path(__file__).resolve().parents[1]/'Queue-Management-System-v2-main/Queue-Management-System-v2-main/ensemble_predict.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cache_allowed')
        from prediction.arrival_calibration import target_definition
        env={'_path_is_fresh':lambda p:True,'target_definition':target_definition}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),env)
        cache=env['_cache_allowed']
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp)/'model';model.write_text('trained')
            meta={'source':'REAL','data_span_days':30,'bootstrap':False,'train_fingerprint':'older-data'}
            self.assertFalse(cache([model],meta,30,'REAL',False,'auto'))
            meta['arrival_target']=target_definition()
            self.assertTrue(cache([model],meta,30,'REAL',False,'auto'))
            self.assertFalse(cache([model],meta,30,'SIM',False,'auto'))
            self.assertFalse(cache([model],meta,7,'REAL',False,'auto'))
            self.assertFalse(cache([model],meta,30,'REAL',False,'train'))
            env['_path_is_fresh']=lambda p:False
            self.assertFalse(cache([model],meta,30,'REAL',False,'auto'))
            self.assertTrue(cache([model],meta,30,'REAL',False,'infer'))
            model.unlink()
            self.assertFalse(cache([model],meta,30,'REAL',False,'infer'))

if __name__=='__main__': unittest.main()
