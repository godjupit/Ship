"""TensorRT 的配置、缓存校验和委派测试不要求安装 TensorRT 或真实 engine。"""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from supership.generate_submission import forward_model
from supership.inference_config import InferenceConfig
from supership.tensorrt_runner import TensorRTRunner, model_fingerprint


class TensorRTTests(unittest.TestCase):
    def test_invalid_backend(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            InferenceConfig(backend="unknown").validate()

    def test_cpu_rejected_without_cuda_or_runtime_import(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            TensorRTRunner("missing.engine", "cpu")

    def test_fingerprint_tracks_weights_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            path.write_bytes(b"one")
            config = {"model": "RTv4", "num_classes": 3, "RTv4": {"backbone": "HGNetv2"},
                      "HGNetv2": {"name": "B0"}, "val_dataloader": {"num_workers": 4}}
            first = model_fingerprint(config, path)
            config["val_dataloader"]["num_workers"] = 8
            self.assertEqual(first, model_fingerprint(config, path))
            config["HGNetv2"]["name"] = "B1"
            self.assertNotEqual(first, model_fingerprint(config, path))
            config["HGNetv2"]["name"] = "B0"
            self.assertEqual(first, model_fingerprint(dict(reversed(list(config.items()))), path))
            path.write_bytes(b"two")
            self.assertNotEqual(first, model_fingerprint(config, path))

    def test_stale_and_corrupt_engine_rejected_before_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.engine"
            path.write_bytes(b"serialized engine")
            meta = {"model_fingerprint": "expected", "tensorrt_version": "test",
                    "engine_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            Path(str(path) + ".json").write_text(json.dumps(meta))
            with patch("supership.tensorrt_runner.load_tensorrt", return_value=SimpleNamespace(__version__="test")), \
                 patch("supership.tensorrt_runner.make_logger", return_value=None):
                with self.assertRaisesRegex(ValueError, "config/weights"):
                    TensorRTRunner(path, "cuda:0", "other")
                path.write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    TensorRTRunner(path, "cuda:0", "expected")

    def test_runner_delegation(self):
        runner = object.__new__(TensorRTRunner)
        expected = {"pred_logits": torch.ones(1), "pred_boxes": torch.ones(1)}
        runner.forward = Mock(return_value=expected)
        samples = torch.ones(1)
        self.assertIs(forward_model(runner, samples), expected)
        runner.forward.assert_called_once_with(samples)
