"""
create_logger in both camera apps: log files rotate instead of growing forever, and
the console only gets warnings when its output is redirected to a file.
"""
import importlib.util
import io
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
APPS = {
    'checkout': ROOT / 'Head-Detector/utils/queue_utils.py',
    'entrance': ROOT / 'Queue-Management-System-v2-main/Queue-Management-System-v2-main/utils/queue_utils.py',
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(f'queue_utils_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TtyStream(io.StringIO):
    def isatty(self):
        return True


class LogRotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.loggers = []

    def tearDown(self):
        for logger in self.loggers:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        self.tmp.cleanup()

    def make(self, module, tag, **env):
        name = f'test_{tag}_{len(self.loggers)}'
        with patch.dict(os.environ, env, clear=False):
            for key in ('LOG_MAX_MB', 'LOG_BACKUP_COUNT', 'LOG_CONSOLE_LEVEL'):
                if key not in env:
                    os.environ.pop(key, None)
            logger = module.create_logger(name=name, level='DEBUG', log_dir=self.tmp.name, file=f'{name}.log')
        self.loggers.append(logger)
        return logger, name

    def test_files_rotate_with_bounded_copies(self):
        for app, path in APPS.items():
            with self.subTest(app=app):
                module = load(app, path)
                with patch.object(sys, 'stderr', io.StringIO()):
                    logger, name = self.make(module, app, LOG_MAX_MB='0.001', LOG_BACKUP_COUNT='2')
                    for i in range(300):
                        logger.info('detection line %d with some padding to fill the file', i)
                files = sorted(p.name for p in Path(self.tmp.name).glob(f'{name}.log*'))
                self.assertEqual(files, [f'{name}.log', f'{name}.log.1', f'{name}.log.2'])
                for p in Path(self.tmp.name).glob(f'{name}.log*'):
                    self.assertLess(p.stat().st_size, 2048)

    def test_console_quiet_when_redirected_full_in_a_window(self):
        for app, path in APPS.items():
            with self.subTest(app=app):
                module = load(app, path)
                with patch.object(sys, 'stderr', io.StringIO()):
                    redirected, _ = self.make(module, app + '_file')
                with patch.object(sys, 'stderr', TtyStream()):
                    window, _ = self.make(module, app + '_tty')
                console = lambda lg: next(h for h in lg.handlers if type(h) is logging.StreamHandler)
                self.assertEqual(console(redirected).level, logging.WARNING)
                self.assertEqual(console(window).level, logging.DEBUG)

    def test_console_level_override(self):
        module = load('checkout', APPS['checkout'])
        with patch.object(sys, 'stderr', io.StringIO()):
            logger, _ = self.make(module, 'override', LOG_CONSOLE_LEVEL='info')
        handler = next(h for h in logger.handlers if type(h) is logging.StreamHandler)
        self.assertEqual(handler.level, logging.INFO)


if __name__ == '__main__':
    unittest.main()
