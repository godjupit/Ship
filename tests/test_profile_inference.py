"""检查模块计时挂钩不会改变模型输出，也不会在结束后残留。"""
import contextlib
import io
import unittest

import torch

from scripts.profile_inference import StageTimings


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.encoder = torch.nn.Linear(4, 4)
        self.decoder = torch.nn.Linear(4, 4)

    def forward(self, samples):
        return self.decoder(self.encoder(self.backbone(samples)))


class ModuleTimingTests(unittest.TestCase):
    def check_timings(self, device):
        model = TinyModel().to(device).eval()
        samples = torch.ones(2, 4, device=device)
        timing = StageTimings(device)
        with torch.inference_mode():
            expected = model(samples)
            with timing.modules(model):
                actual = model(samples)
        torch.testing.assert_close(actual, expected)
        if device == "cuda:0":
            torch.cuda.synchronize(device)
        with contextlib.redirect_stderr(io.StringIO()):
            values = timing.report()
        suffix = "_gpu_s" if device == "cuda:0" else "_wall_s"
        for name in ("backbone", "encoder", "decoder"):
            self.assertGreater(values[name + suffix], 0)
            self.assertFalse(getattr(model, name)._forward_hooks)
            self.assertFalse(getattr(model, name)._forward_pre_hooks)

    def test_cpu(self):
        self.check_timings("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda(self):
        self.check_timings("cuda:0")

    def test_exception_removes_hooks(self):
        model = TinyModel()
        with self.assertRaises(RuntimeError):
            with StageTimings("cpu").modules(model):
                model(torch.ones(2, 9))
        for module in (model.backbone, model.encoder, model.decoder):
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
