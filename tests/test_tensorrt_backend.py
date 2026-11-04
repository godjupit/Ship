"""Check static-engine graph replay and padding without evaluation data."""

from pathlib import Path
import unittest

import torch

from supership.backends import TensorRTRuntime


@unittest.skipUnless(torch.cuda.is_available() and Path("weights/supership-544.engine").is_file(),
                     "requires CUDA and the locally built 544 TensorRT engine")
class TensorRTBackendTests(unittest.TestCase):
    def test_graph_reads_updated_input_and_tail_matches_padded_batch(self):
        runtime = TensorRTRuntime("weights/supership-544.engine", 8, "cuda:0", "weights/baseline.pth")
        images = torch.zeros((8, 3, 544, 544), device="cuda:0")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                runtime(images)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
            captured = runtime(images)
        with torch.inference_mode():
            images.uniform_(0, 1)
            graph.replay()
            torch.cuda.synchronize()
            actual = {name: value.clone() for name, value in captured.items()}
            expected = runtime(images)
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

            tail = {name: value.clone() for name, value in runtime(images[:3]).items()}
            padded = torch.zeros_like(images)
            padded[:3].copy_(images[:3])
            full = runtime(padded)
            for name in tail:
                self.assertEqual(tail[name].shape[0], 3)
                torch.testing.assert_close(tail[name], full[name][:3], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
