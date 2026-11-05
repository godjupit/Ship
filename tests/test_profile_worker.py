"""The profiling request must stop/flush even when inference fails."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.experiments import profile_worker


class ProfileWorkerTests(unittest.TestCase):
    def run_profile(self, fail=False):
        events = []

        def record(name):
            def call():
                events.append(name)
                return (0,)
            return call

        runtime = SimpleNamespace(cu=SimpleNamespace(
            cuProfilerStart=record('start'), cuProfilerStop=record('stop')))
        request = {'meta_file': 'request.json'}

        def inference(actual_runtime, actual_request):
            self.assertIs(actual_runtime, runtime)
            self.assertIs(actual_request, request)
            events.append('request')
            if fail:
                raise ValueError('bad image')
            return 27, 9

        with patch.object(profile_worker, 'original_run', inference):
            if fail:
                with self.assertRaisesRegex(ValueError, 'bad image'):
                    profile_worker.profiled_run(runtime, request)
            else:
                self.assertEqual(profile_worker.profiled_run(runtime, request), (27, 9))
        self.assertEqual(events, ['start', 'request', 'stop'])

    def test_request_preserves_result_and_flushes(self):
        self.run_profile()

    def test_failed_request_still_flushes(self):
        self.run_profile(fail=True)
