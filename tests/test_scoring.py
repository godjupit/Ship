import copy
import unittest

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

from supership.score_submission import load_detections


class ScoringTests(unittest.TestCase):
    def ground_truth(self, crowd=False):
        gt = COCO()
        gt.dataset = {
            "images": [{"id": 7, "width": 100, "height": 100}],
            "categories": [{"id": 0, "name": "ship"}],
            "annotations": [{"id": 1, "image_id": 7, "category_id": 0,
                             "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
        }
        if crowd:
            gt.dataset["annotations"].append({
                "id": 2, "image_id": 7, "category_id": 0,
                "bbox": [0, 0, 100, 100], "area": 10000, "iscrowd": 1,
            })
        gt.createIndex()
        return gt

    def detection(self, bbox=None, score=0.5, **kwargs):
        return {"image_id": 7, "category_id": 0,
                "bbox": bbox if bbox is not None else [10, 10, 20, 20],
                "score": score, **kwargs}

    def evaluate(self, predictions, crowd=False):
        gt = self.ground_truth(crowd)
        evaluator = COCOeval_faster(
            gt, cocoDt=load_detections(gt, predictions), iouType="bbox",
            print_function=lambda *args: None,
        )
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return evaluator

    def test_nonpositive_boxes_count_as_false_positives(self):
        for bbox in [[10, 10, -20, 20], [10, 10, 20, -20],
                     [30, 30, -20, -20], [10, 10, 0, 20], [10, 10, 20, 0]]:
            for crowd in [False, True]:
                with self.subTest(bbox=bbox, crowd=crowd):
                    predictions = [self.detection(bbox, 0.9), self.detection()]
                    original = copy.deepcopy(predictions)
                    result = self.evaluate(predictions, crowd)
                    self.assertAlmostEqual(result.stats[1], 0.5, places=6)
                    self.assertTrue(np.all(result.ious[7, 0][0] == 0))
                    self.assertEqual(predictions, original)

    def test_top100_retains_invalid_detections(self):
        predictions = [self.detection([0, 0, -1, 1], .9) for _ in range(100)]
        result = self.evaluate(predictions + [self.detection()])
        self.assertEqual(result.stats[1], 0)

    def test_only_invalid_and_empty_outputs(self):
        for predictions in [[], [self.detection([0, 0, -1, 1])]]:
            self.assertEqual(self.evaluate(predictions).stats[1], 0)

    def test_extra_fields_cannot_hide_false_positive(self):
        predictions = [self.detection(
            [30, 30, -20, -20], .9, drop=True, ignore=True,
            area=-1, id=999, iscrowd=1, segmentation=[],
        ), self.detection()]
        self.assertAlmostEqual(self.evaluate(predictions).stats[1], 0.5, places=6)

    def test_valid_results_match_standard_coco(self):
        predictions = [self.detection(), self.detection([50, 50, 20, 20], .2)]
        gt = self.ground_truth()
        standard = COCOeval_faster(
            gt, cocoDt=gt.loadRes(copy.deepcopy(predictions)), iouType="bbox",
            print_function=lambda *args: None,
        )
        standard.evaluate()
        standard.accumulate()
        actual = self.evaluate(predictions)
        np.testing.assert_array_equal(actual.eval["precision"], standard.eval["precision"])


if __name__ == "__main__":
    unittest.main()
