"""验证跨流依赖、CPU 源数据保活、GPU 显存生命周期和 Graph 兼容性。"""
import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from supership.cuda_graph_runner import CUDAGraphRunner
from supership.prefetch import DevicePrefetcher


def batches(shape=(3, 32, 32), pinned=False):
    for count, value in [(2, 1.), (2, 2.), (1, 3.), (2, 4.)]:
        samples = torch.full((count, *shape), value, pin_memory=pinned)
        targets = [{"orig_size": torch.tensor([int(value), 10]),
                    "image_id": torch.tensor([int(value)])} for _ in range(count)]
        yield samples, targets


class PrefetchTests(unittest.TestCase):
    def test_cpu_and_disabled_keep_values_and_partial_batch(self):
        for enabled in (False, True):
            with DevicePrefetcher(batches(), "cpu", enabled) as loader:
                observed = []
                for samples, targets, sizes in loader:
                    observed.append((len(samples), samples[0, 0, 0, 0].item()))
                    self.assertEqual(sizes.tolist(), [t["orig_size"].tolist() for t in targets])
                    loader.prefetch_next()
            self.assertEqual(observed, [(2, 1.), (2, 2.), (1, 3.), (2, 4.)])

    def test_empty(self):
        with DevicePrefetcher([], "cpu", True) as loader:
            self.assertEqual(list(loader), [])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cross_stream_values_survive_allocator_pressure(self):
        stream = torch.cuda.Stream()
        snapshots = []
        with torch.inference_mode(), torch.cuda.stream(stream), \
             patch("torch.cuda.synchronize", side_effect=AssertionError("global sync")), \
             DevicePrefetcher(batches(pinned=True), "cuda:0", True) as loader:
            self.assertNotEqual(loader.stream.cuda_stream, stream.cuda_stream)
            for samples, targets, sizes in loader:
                torch.cuda._sleep(5_000_000)
                snapshots.append((samples.square(), sizes + 1))
                loader.prefetch_next()
                del samples, sizes
                # 在分配它的拷贝流制造显存复用压力，消费流仍可能在读取上一批。
                with torch.cuda.stream(loader.stream):
                    scratch = torch.empty((2, 3, 32, 32), device="cuda")
                    scratch.fill_(-99)
        stream.synchronize()
        for (samples, sizes), (count, value) in zip(snapshots, [(2, 1), (2, 2), (1, 3), (2, 4)]):
            torch.testing.assert_close(samples.cpu(), torch.full((count, 3, 32, 32), float(value**2)))
            torch.testing.assert_close(sizes.cpu(), torch.tensor([[value + 1, 11]] * count))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cpu_source_retained_until_copy_and_early_close(self):
        source = torch.ones((2, 3, 32, 32), pin_memory=True)
        source_ref = weakref.ref(source)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
        # 先阻塞拷贝流，确认 DMA 未完成时引用不会被释放。
        iterable = iter([(source, [{"orig_size": torch.tensor([1, 2])}] * 2)])
        with patch("torch.cuda.Stream", return_value=stream):
            loader = DevicePrefetcher(iterable, "cuda:0", True)
        next(loader.iterator, None)  # 释放一次性迭代器持有的源数据，单独检验保活引用。
        del source, iterable
        gc.collect()
        self.assertIsNotNone(source_ref())
        loader.close()
        self.assertFalse(loader.inflight)
        gc.collect()
        self.assertIsNone(source_ref())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_and_partial_batch_fallback(self):
        def eager(samples):
            return samples.mean(dim=(2, 3))

        runner = CUDAGraphRunner(eager, "cuda:0", 2)
        snapshots = []
        with torch.inference_mode(), DevicePrefetcher(
            batches((3, 640, 640), pinned=True), "cuda:0", True
        ) as loader:
            for samples, targets, sizes in loader:
                snapshots.append(runner.forward(samples).clone())
                loader.prefetch_next()
        for result, (count, value) in zip(snapshots, [(2, 1.), (2, 2.), (1, 3.), (2, 4.)]):
            torch.testing.assert_close(result.cpu(), torch.full((count, 3), value))
