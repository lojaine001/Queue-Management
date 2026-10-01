import subprocess
import unittest
from unittest.mock import Mock, patch

from prediction.dashboard_lstm import forecast_dwell


class ForecastWorkerTests(unittest.TestCase):
    def call_worker(self):
        return forecast_dwell([1.0] * 12, 4, 2, 1, 0.5, 10.0, timeout=3)

    @patch("prediction.dashboard_lstm.subprocess.Popen")
    def test_returns_complete_finite_forecast(self, popen):
        popen.return_value = Mock(returncode=0)
        popen.return_value.communicate.return_value = ('[1.5, 2.0]', '')
        self.assertEqual(self.call_worker(), [1.5, 2.0])

    @patch("prediction.dashboard_lstm.subprocess.Popen")
    def test_rejects_incomplete_or_invalid_worker_output(self, popen):
        popen.return_value = Mock(returncode=0)
        for output in ('[1.5]', '[NaN, 2]', '[1, 100]', '["1", 2]', '{}'):
            with self.subTest(output=output):
                popen.return_value.communicate.return_value = (output, '')
                with self.assertRaises(ValueError):
                    self.call_worker()

    @patch("prediction.dashboard_lstm.subprocess.Popen")
    def test_failed_training_propagates_for_dashboard_fallback(self, popen):
        popen.return_value = Mock(returncode=1)
        popen.return_value.communicate.return_value = ('', 'training failed')
        with self.assertRaisesRegex(RuntimeError, 'training failed'):
            self.call_worker()

    @patch("prediction.dashboard_lstm.subprocess.run")
    @patch("prediction.dashboard_lstm.subprocess.Popen")
    def test_windows_timeout_terminates_worker_and_launcher(self, popen, run):
        process = popen.return_value
        process.pid = 1234
        process.communicate.side_effect = [
            subprocess.TimeoutExpired('worker', 3), ('', ''),
        ]
        # Patch after Popen setup; this also works when tests run on Linux.
        with patch("prediction.dashboard_lstm.os.name", "nt"):
            with patch("prediction.dashboard_lstm.Path") as path:
                path.return_value.resolve.return_value = 'worker.py'
                with self.assertRaises(TimeoutError):
                    self.call_worker()
        self.assertEqual(run.call_args.args[0], ['taskkill', '/PID', '1234', '/T', '/F'])
        self.assertEqual(process.communicate.call_count, 2)


if __name__ == '__main__':
    unittest.main()
