"""验证 AP 比较使用真实 COCO 评分，且差值方向正确。"""
import json
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from supership.compare_reference import score_predictions
from supership.benchmark_submission import main as benchmark_main


class AccuracyComparisonTests(unittest.TestCase):
    def test_benchmark_adds_only_requested_ap50_line(self):
        args = SimpleNamespace(runs=1, accuracy_gt="gt.json")
        captured = io.StringIO()
        with patch("supership.benchmark_submission.parse_args", return_value=args), \
             patch("supership.benchmark_submission.run_once", return_value=(1.25, "result.json", 3)), \
             patch("supership.compare_reference.score_predictions", return_value={"AP50": 0.79}), \
             contextlib.redirect_stdout(captured):
            benchmark_main()
        lines = captured.getvalue().splitlines()
        self.assertEqual(lines[0], "run 1/1: 1.250000 s, 3 detections, result.json")
        self.assertEqual(lines[1], "AP50 / baseline = 0.790000 / 0.793473")
        self.assertEqual(json.loads(lines[2]), {"runs_seconds": [1.25], "median_seconds": 1.25})
        self.assertEqual(len(lines), 3)

    def test_perfect_and_missing_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            gt = Path(directory) / "gt.json"
            result = Path(directory) / "result.json"
            gt.write_text(json.dumps({
                "images": [{"id": 1, "width": 100, "height": 100}],
                "categories": [{"id": 0, "name": "ship"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                                 "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
            }))
            result.write_text(json.dumps([{"image_id": 1, "category_id": 0,
                                           "bbox": [10, 10, 20, 20], "score": 1.0}]))
            perfect = score_predictions(result, gt)
            result.write_text("[]")
            missing = score_predictions(result, gt)
            for name in ("mAP", "AP50", "AP75"):
                self.assertAlmostEqual(perfect[name], 1)
                self.assertEqual(missing[name], 0)
