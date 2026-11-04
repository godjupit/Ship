"""验证 AP 比较使用真实 COCO 评分，且差值方向正确。"""
import json
from pathlib import Path
import tempfile
import unittest

from supership.compare_reference import (
    REFERENCE_BASELINE, accuracy_difference, reference_accuracy_check, score_predictions,
)


class AccuracyComparisonTests(unittest.TestCase):
    def test_reference_threshold_boundary(self):
        threshold = 0.98 * REFERENCE_BASELINE["AP50"]
        self.assertTrue(reference_accuracy_check({"AP50": threshold})["passed"])
        self.assertFalse(reference_accuracy_check({"AP50": threshold - 1e-8})["passed"])
        # 验证集 0.916 的阈值不能用来判定 200 张参考集。
        self.assertLess(threshold, 0.8)

    def test_difference_direction(self):
        delta = accuracy_difference({"mAP": 0.6, "AP50": 0.9, "AP75": 0.8},
                                    {"mAP": 0.7, "AP50": 0.9, "AP75": 0.7})
        self.assertAlmostEqual(delta["mAP"], -0.1)
        self.assertEqual(delta["AP50"], 0)
        self.assertAlmostEqual(delta["AP75"], 0.1)

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
