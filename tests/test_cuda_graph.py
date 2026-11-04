"""Integration checks: run with .venv/bin/python -m unittest discover -s tests."""

import unittest
import gc
from pathlib import Path

import torch

from supership.backends import CUDAGraphRuntime, TensorRTRuntime
from supership.generate_submission import build_runtime, model_forward


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class CUDAGraphTests(unittest.TestCase):
    def test_full_and_tail_batches(self):
        _, model, _ = build_runtime("configs/dfine/dfine_hgnetv2_s_supership.yml",
                                    "weights/baseline.pth", "cuda:0")
        runtime = CUDAGraphRuntime(model, 8, "cuda:0")
        with torch.inference_mode():
            for count in (8, 6, 1):
                with self.subTest(count=count):
                    images = torch.rand(count, 3, 640, 640, device="cuda")
                    padded = torch.zeros(8, 3, 640, 640, device="cuda")
                    padded[:count].copy_(images)
                    expected = model_forward(model, padded, "fp32")
                    actual = runtime(images)
                    # Compare the same padded batch; changing batch shape can
                    # reorder near-tied TopK proposals even in eager execution.
                    for key in ("pred_logits", "pred_boxes"):
                        torch.testing.assert_close(actual[key], expected[key][:count],
                                                   rtol=1e-4, atol=1e-4)
            with self.assertRaises(ValueError):
                runtime(torch.zeros(9, 3, 640, 640, device="cuda"))

    @unittest.skipUnless(Path("output/deploy/model.fp16.engine").exists(), "Build FP16 engine first")
    def test_tensorrt_capture_keeps_context_alive(self):
        def capture():
            engine = TensorRTRuntime("output/deploy/model.fp16.engine", 8, "cuda:0", "weights/baseline.pth")
            engine.is_runtime = True
            return CUDAGraphRuntime(engine, 8, "cuda:0")

        captured = capture()
        gc.collect()
        reference = TensorRTRuntime("output/deploy/model.fp16.engine", 8, "cuda:0", "weights/baseline.pth")
        with torch.inference_mode():
            for count in (8, 6, 1):
                with self.subTest(count=count):
                    images = torch.rand(count, 3, 640, 640, device="cuda")
                    padded = torch.zeros(8, 3, 640, 640, device="cuda")
                    padded[:count].copy_(images)
                    expected = reference(padded)
                    actual = captured(images)
                    for key in ("pred_logits", "pred_boxes"):
                        torch.testing.assert_close(actual[key], expected[key][:count], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
