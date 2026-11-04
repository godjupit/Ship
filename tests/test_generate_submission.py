import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from supership.inference_config import INFERENCE_CONFIG, InferenceConfig

from supership.generate_submission import amp_enabled, forward_model, prepare_cuda_graph, run_inference


class GraphTests(unittest.TestCase):
    def test_disabled_keeps_original_model(self):
        model = torch.nn.Identity()
        with patch.object(INFERENCE_CONFIG, "cuda_graph", False), \
             patch("torch.cuda.CUDAGraph", side_effect=AssertionError("unexpected capture")):
            self.assertIs(prepare_cuda_graph(model, "cuda:0", 2), model)

    def test_cpu_does_not_capture(self):
        model = torch.nn.Identity()
        with patch.object(INFERENCE_CONFIG, "cuda_graph", True):
            self.assertIs(prepare_cuda_graph(model, "cpu", 2), model)
        self.assertFalse(hasattr(model, "_cuda_graph_runner"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_replay_updates_inputs_and_partial_batch_falls_back(self):
        class Model(torch.nn.Module):
            def forward(self, samples):
                pooled = samples.mean(dim=(2, 3))
                return {"pred_logits": pooled * 2, "pred_boxes": pooled + 1}

        model = Model().cuda().eval()
        with patch.object(INFERENCE_CONFIG, "cuda_graph", True), patch.object(INFERENCE_CONFIG, "amp", False):
            runner = prepare_cuda_graph(model, "cuda:0", 2)
        with torch.inference_mode():
            pointer = None
            for count, value in [(2, 1.), (2, 3.), (1, 7.)]:
                samples = torch.full((count, 3, 640, 640), value, device="cuda:0")
                expected = model(samples)
                actual = runner.forward(samples)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key])
                if count == 2:
                    if pointer is None:
                        pointer = actual["pred_logits"].data_ptr()
                    self.assertEqual(actual["pred_logits"].data_ptr(), pointer)
                else:
                    self.assertEqual(actual["pred_logits"].shape[0], 1)


class AMPTests(unittest.TestCase):
    def test_reserved_options_fail_explicitly(self):
        for name in ("channels_last", "fuse_conv_bn"):
            with self.assertRaisesRegex(NotImplementedError, name):
                InferenceConfig(**{name: True}).validate()

    def test_switch_defaults_off_and_cpu_stays_fp32(self):
        with patch.object(INFERENCE_CONFIG, "amp", False):
            self.assertFalse(amp_enabled("cuda:0"))
        with patch.object(INFERENCE_CONFIG, "amp", True):
            self.assertTrue(amp_enabled("cuda:0"))
            self.assertFalse(amp_enabled("cpu"))

    def test_disabled_preserves_original_output(self):
        outputs = {"pred_logits": torch.ones(1), "pred_boxes": torch.ones(1)}
        self.assertIs(forward_model(lambda samples: outputs, torch.ones(1)), outputs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_forward_uses_half_and_returns_float_postprocessing_inputs(self):
        layer = torch.nn.Linear(4, 4).cuda().eval()
        observed = []

        def model(samples):
            result = layer(samples)
            observed.append(result.dtype)
            return {"pred_logits": result, "pred_boxes": result, "extra": 7}

        with torch.inference_mode():
            outputs = forward_model(model, torch.ones(2, 4, device="cuda"), True)
        self.assertEqual(observed, [torch.float16])
        self.assertEqual(outputs["pred_logits"].dtype, torch.float32)
        self.assertEqual(outputs["pred_boxes"].dtype, torch.float32)
        self.assertEqual(outputs["extra"], 7)


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
