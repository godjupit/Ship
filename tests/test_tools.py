import unittest
from types import SimpleNamespace

import numpy as np

from supership.score_submission import per_class_ap
from supership.visualize_annotations import find_image_entry


class ToolTests(unittest.TestCase):
    def test_prediction_array_requires_image_id(self):
        self.assertEqual(find_image_entry([], "image.jpg", 42), 42)
        with self.assertRaises(SystemExit):
            find_image_entry([], "image.jpg", None)

    def test_coco_image_lookup(self):
        self.assertEqual(find_image_entry(
            {"images": [{"file_name": "nested/image.jpg", "id": 7}]}, "image.jpg", None
        ), 7)

    def test_per_class_ap_has_no_global_state_dependency(self):
        evaluator = SimpleNamespace(
            eval={"precision": np.ones((1, 1, 1, 1, 1))},
            params=SimpleNamespace(catIds=[0]),
            cocoGt=SimpleNamespace(loadCats=lambda ids: [{"id": 0, "name": "carrier"}]),
        )
        self.assertEqual(per_class_ap(evaluator), {"0:carrier": 1.0})


if __name__ == "__main__":
    unittest.main()
