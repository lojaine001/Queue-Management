import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import psutil
import run_scheduler as scheduler
from prediction.runtime import RunLock, AlreadyRunning, write_json, read_json
from prediction.scheduler_control import owned_process, terminate_tree


class SchedulerTests(unittest.TestCase):
    def test_lock_excludes_second_process_and_releases_after_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / 'owner.lock'
            code = 'from prediction.runtime import RunLock; import time; l=RunLock(' + repr(str(lock)) + '); l.__enter__(); print("ready",flush=True); time.sleep(60)'
            p = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                self.assertEqual(p.stdout.readline().strip(), 'ready')
                with self.assertRaises(AlreadyRunning):
                    with RunLock(lock): pass
            finally:
                terminate_tree(p.pid, psutil.Process(p.pid).create_time())
                p.wait(timeout=10)
                p.stdout.close()
            with RunLock(lock): pass

    def test_pid_reuse_is_rejected(self):
        self.assertIsNone(owned_process(psutil.Process().pid, psutil.Process().create_time()-100))

    def test_deadline_skips_missed_slots(self):
        self.assertEqual(scheduler.next_deadline(100, 180, 674), 820)
        self.assertEqual(scheduler.next_deadline(100, 180, 110), 280)

    def test_arguments_reject_nonpositive(self):
        for value in ['0', '-3']:
            with self.assertRaises(argparse.ArgumentTypeError): scheduler.positive_int(value)

    def run_dummy(self, script, timeout=10, stop=False):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predict.py'
            path.write_text(script, encoding='utf-8')
            state = {'owner': 'test'}
            snapshots = []
            args = argparse.Namespace(source='REAL', days=30, timeout=timeout)
            with patch.object(scheduler, 'RUNTIME', Path(tmp)), patch.object(scheduler, 'PREDICT_SCRIPT', path):
                scheduler.run_job(args, state, lambda: snapshots.append(dict(state)), lambda: stop)
            return state, snapshots

    def test_success_only_after_exit(self):
        state, snaps = self.run_dummy('print("published")')
        self.assertEqual(state['status'], 'waiting')
        self.assertIn('last_success', state)
        self.assertNotIn('last_success', snaps[0])

    def test_failure_records_diagnostic(self):
        state, _ = self.run_dummy('raise RuntimeError("deliberate failure")')
        self.assertEqual(state['status'], 'failed')
        self.assertIn('deliberate failure', state['last_error'])
        self.assertNotIn('last_success', state)

    def test_graceful_stop_waits_for_active_job(self):
        state, snaps = self.run_dummy('import time; time.sleep(1.5); print("finished")', stop=True)
        self.assertTrue(any(s.get('status') == 'stopping' for s in snaps))
        self.assertEqual(state['exit_code'], 0)

    def test_timeout_kills_nested_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'pid'
            child_code = 'import os,time; from pathlib import Path; Path(' + repr(str(marker)) + ').write_text(str(os.getpid())); time.sleep(60)'
            script = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",' + repr(child_code) + '], creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)); time.sleep(60)'
            state, _ = self.run_dummy(script, timeout=3)
            self.assertEqual(state['status'], 'failed')
            self.assertIn('deadline', state['last_error'])
            self.assertTrue(marker.exists())
            self.assertFalse(psutil.pid_exists(int(marker.read_text())))

    def test_atomic_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.json'
            write_json(path, {'pid': 123})
            self.assertEqual(read_json(path), {'pid':123})
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

if __name__ == '__main__': unittest.main()
