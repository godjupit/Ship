import subprocess
import sys
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from supership.benchmark_submission import run_once, send_request, terminate, validate_output, wait_for_ready


class OutputValidationTests(unittest.TestCase):
    def validate(self, detections, metadata=None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.json"
            output.write_text(json.dumps(detections))
            if metadata is None:
                return validate_output(output)
            meta = Path(directory) / "meta.json"
            meta.write_text(json.dumps(metadata))
            return validate_output(output, meta)

    def detection(self, **updates):
        detection = {"image_id": 7, "category_id": 0, "bbox": [-1, 0, 3, 4], "score": 0.5}
        detection.update(updates)
        return detection

    def test_invalid_ids(self):
        for value in ["7", 7.0, True, None, [], {}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate([self.detection(image_id=value)])

    def test_unknown_image(self):
        with self.assertRaisesRegex(ValueError, "unknown image_id"):
            self.validate([self.detection()], {"images": [{"id": 8}]})

    def test_invalid_category_types(self):
        for value in [True, 1.0, "1", 3]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate([self.detection(category_id=value)])

    def test_invalid_numbers(self):
        for updates in [{"score": True}, {"score": float("nan")},
                        {"bbox": [0, 0, float("inf"), 1]}, {"bbox": [False, 0, 1, 1]},
                        {"bbox": [0, 0, 1e300, 1e300]}, {"score": 10 ** 1000}]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.validate([self.detection(**updates)])

    def test_nonpositive_dimensions_are_accepted(self):
        for bbox in [[0, 0, 0, 1], [0, 0, 1, -1], [0, 0, -1, -1]]:
            self.assertEqual(self.validate([self.detection(bbox=bbox)]), 1)

    def test_metadata_validation(self):
        for meta in [[], {}, {"images": [{}]}, {"images": [{"id": True}]},
                     {"images": [{"id": 7}, {"id": 7}]}]:
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                self.validate([], meta)

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.json"
            output.write_text('[{"image_id":1,"image_id":2}]')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                validate_output(output)

    def test_nonfinite_extra_field(self):
        with self.assertRaises(ValueError):
            self.validate([self.detection(extra=float("nan"))])

    def test_overflow_in_extra_field(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.json"
            output.write_text(json.dumps([self.detection(extra="OVERFLOW")]).replace('"OVERFLOW"', '1e999'))
            with self.assertRaises(ValueError):
                validate_output(output)

    def test_empty_output_and_images_without_detections_are_legal(self):
        meta = {"images": [{"id": 7}, {"id": 8}]}
        self.assertEqual(self.validate([], meta), 0)
        self.assertEqual(self.validate([self.detection()], meta), 1)


class ReadyProtocolTests(unittest.TestCase):
    def worker(self, code):
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.addCleanup(process.stdout.close)
        self.addCleanup(process.stdin.close)
        self.addCleanup(terminate, process)
        return process

    def test_ready(self):
        process = self.worker(
            "import sys; print('THU_RS_READY', flush=True); sys.stdin.readline()"
        )
        wait_for_ready(process, 1)

    def test_ready_after_batched_logs(self):
        process = self.worker(
            "import os, sys; "
            "os.write(1, b'initializing\\nmodel loaded\\nTHU_RS_READY\\n'); "
            "sys.stdin.readline()"
        )
        wait_for_ready(process, 1)

    def test_fragmented_ready(self):
        process = self.worker(
            "import os, sys, time; os.write(1, b'THU_RS_'); "
            "time.sleep(0.05); os.write(1, b'READY\\n'); sys.stdin.readline()"
        )
        wait_for_ready(process, 1)

    def test_partial_log_does_not_disable_timeout(self):
        process = self.worker(
            "import os, time; os.write(1, b'loading'); time.sleep(1)"
        )
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            wait_for_ready(process, 0.2)
        self.assertLess(time.monotonic() - start, 0.8)

    def test_exit_before_ready(self):
        process = self.worker("pass")
        with self.assertRaisesRegex(RuntimeError, 'before READY'):
            wait_for_ready(process, 1)

    def test_binary_and_long_init_logs(self):
        process = self.worker(
            "import os,sys; os.write(1,b'\\xff'*200000); "
            "os.write(1,b'\\nTHU_RS_READY\\n'); sys.stdin.readline()"
        )
        wait_for_ready(process, 2)

    def test_request_timeout(self):
        process = self.worker("import time; time.sleep(10)")
        with self.assertRaisesRegex(TimeoutError, "sending request"):
            send_request(process.stdin, "x" * 1000000, time.monotonic() + 0.1)


class WorkerLifecycleTests(unittest.TestCase):
    def run_worker(self, code, timeout=2):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        meta = root / "meta.json"
        meta.write_text('{"images": []}')
        args = SimpleNamespace(
            command=[sys.executable, "-c", code], image_dir=str(root), meta_file=str(meta),
            output_file=str(root / "result.json"), runs=1, init_timeout=2, run_timeout=timeout,
        )
        return run_once(args, 0)

    def test_protocol_and_binary_logs(self):
        elapsed, _, count = self.run_worker(
            "import os,sys,json,pathlib; "
            "os.write(1,b'log\\nTHU_RS_READY\\n'); "
            "r=json.loads(sys.stdin.readline()); "
            "os.write(1,b'\\xff'*100000); "
            "pathlib.Path(r['output_file']).write_text('[]')"
        )
        self.assertGreater(elapsed, 0)
        self.assertEqual(count, 0)

    def test_nonzero_exit(self):
        with self.assertRaisesRegex(RuntimeError, "exit code 3"):
            self.run_worker("import sys; print('THU_RS_READY',flush=True); sys.stdin.readline(); sys.exit(3)")

    def test_missing_output(self):
        with self.assertRaises(FileNotFoundError):
            self.run_worker("import sys; print('THU_RS_READY',flush=True); sys.stdin.readline()")

    def test_run_timeout(self):
        with self.assertRaisesRegex(TimeoutError, "inference exceeded"):
            self.run_worker(
                "import sys,time; print('THU_RS_READY',flush=True); sys.stdin.readline(); time.sleep(10)",
                timeout=0.1,
            )

    def test_cleanup_reaches_child_after_parent_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            child = "import time; time.sleep(30)"
            code = (
                "import sys,subprocess,pathlib; "
                f"p=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.DEVNULL); "
                f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); "
                "sys.exit(2)"
            )
            with self.assertRaises(RuntimeError):
                self.run_worker(code)
            pid = int(pid_file.read_text())
            status = Path(f"/proc/{pid}/stat")
            deadline = time.monotonic() + 1
            while status.exists() and time.monotonic() < deadline:
                if status.read_text().split()[2] == "Z":
                    break
                time.sleep(0.01)
            if status.exists():
                self.assertEqual(status.read_text().split()[2], "Z")

    def test_background_writer_is_waited_and_timed(self):
        child = (
            "import sys,time,pathlib; time.sleep(.3); "
            "pathlib.Path(sys.argv[1]).write_text('[]')"
        )
        code = (
            "import sys,subprocess,json,pathlib; "
            "print('THU_RS_READY',flush=True); r=json.loads(sys.stdin.readline()); "
            f"subprocess.Popen([sys.executable,'-c',{child!r},r['output_file']])"
        )
        elapsed, _, count = self.run_worker(code)
        self.assertGreaterEqual(elapsed, 0.3)
        self.assertEqual(count, 0)

    def test_hanging_descendant_is_killed_after_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
            code = (
                "import sys,subprocess,json,pathlib; "
                "print('THU_RS_READY',flush=True); r=json.loads(sys.stdin.readline()); "
                f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
                f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); "
                "pathlib.Path(r['output_file']).write_text('[]')"
            )
            with self.assertRaisesRegex(TimeoutError, "subprocesses still running"):
                self.run_worker(code, timeout=0.3)
            status = Path(f"/proc/{int(pid_file.read_text())}/stat")
            if status.exists():
                self.assertEqual(status.read_text().split()[2], "Z")

    def test_early_exit_with_inherited_stdout_fails_promptly(self):
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "before READY"):
            self.run_worker(
                "import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); sys.exit(2)"
            )
        self.assertLess(time.monotonic() - start, 1.5)


if __name__ == '__main__':
    unittest.main()
