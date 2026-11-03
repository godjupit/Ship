import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from supership.generate_submission import run_inference


class SerializationTests(unittest.TestCase):
    def test_preserves_negative_zero_and_rounded_zero_boxes(self):
        detection = {
            "boxes": torch.tensor([
                [0, 0, -1, 2], [0, 0, 2, -1], [0, 0, -1, -1],
                [0, 0, 0, 2], [0, 0, 0.004, 2], [0, 0, 2, 0.004],
                [-1, -2, 3, 4],
            ]),
            "scores": torch.full((7,), 0.5),
            "labels": torch.zeros(7, dtype=torch.int64),
        }
        target = {"image_id": torch.tensor([7]), "orig_size": torch.tensor([10, 10])}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.json"
            with patch("supership.generate_submission.create_loader",
                       return_value=[(torch.zeros(1, 3, 4, 4), [target])]):
                count = run_inference(None, lambda x: x, lambda *args: [detection],
                                      "cpu", 1, "images", "meta", output)
            self.assertEqual(count, 7)
            self.assertEqual([d["bbox"] for d in json.loads(output.read_text())], [
                [0, 0, -1, 2], [0, 0, 2, -1], [0, 0, -1, -1],
                [0, 0, 0, 2], [0, 0, 0, 2], [0, 0, 2, 0], [-1, -2, 4, 6],
            ])

    def test_matches_original_serialization_including_partial_batch(self):
        detections = [
            {
                "boxes": torch.tensor([[1.234, -2.345, 10.678, 21.234]]),
                "scores": torch.tensor([0.923456]),
                "labels": torch.tensor([label]),
            }
            for label in range(3)
        ]
        image_ids = [7, 1003, 42]
        targets = [
            {"image_id": torch.tensor([image_id]), "orig_size": torch.tensor([1024, 768])}
            for image_id in image_ids
        ]
        loader = [
            (torch.zeros(2, 3, 4, 4), targets[:2]),
            (torch.zeros(1, 3, 4, 4), targets[2:]),
        ]
        expected = []
        # Preserve the original per-box conversion and Python rounding exactly.
        for image_id, detection in zip(image_ids, detections):
            for index, score in enumerate(detection["scores"].tolist()):
                x1, y1, x2, y2 = detection["boxes"][index].tolist()
                expected.append({
                    "image_id": image_id,
                    "category_id": int(detection["labels"][index]),
                    "bbox": [round(x1, 2), round(y1, 2),
                             round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(score, 5),
                })

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "submission.json"
            with patch("supership.generate_submission.create_loader", return_value=loader):
                with patch("torch.nn.Identity.forward", side_effect=[detections[:2], detections[2:]]):
                    count = run_inference(
                        None, lambda samples: samples, torch.nn.Identity(), "cpu", 2,
                        "unused_images", "unused_meta", output,
                    )
            self.assertEqual(count, 3)
            self.assertEqual(
                output.read_text(), json.dumps(expected, separators=(",", ":")),
            )

    def test_empty_input_writes_json_array(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.json"
            with patch("supership.generate_submission.create_loader", return_value=[]):
                count = run_inference(None, None, None, "cpu", 8, "images", "meta", output)
            self.assertEqual(count, 0)
            self.assertEqual(output.read_text(), "[]")


if __name__ == '__main__':
    unittest.main()
