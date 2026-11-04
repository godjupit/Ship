"""单批 H2D 预取；不复用 CPU 数据缓冲区，不触碰模型或 CUDA Graph 输出。"""
from contextlib import nullcontext

import torch


class DevicePrefetcher:
    """下一批用拷贝流上传，消费批次时只让计算流等待该批的完成事件。"""

    def __init__(self, loader, device, enabled=False, section=None):
        self.device = torch.device(device)
        self.enabled = enabled and self.device.type == "cuda"
        self.section = section or (lambda *args, **kwargs: nullcontext())
        self.stream = torch.cuda.Stream(device=self.device) if self.enabled else None
        self.pending = None
        self.inflight = []
        self.exhausted = False
        with self.section("loader_wait"):
            self.iterator = iter(loader)
        if self.enabled:
            self.prefetch_next()

    def _next_cpu_batch(self):
        """沿用 DataLoader 的 worker 预取，CPU 图像和 targets 不转交给计算流修改。"""
        with self.section("loader_wait"):
            batch = next(self.iterator, None)
        if batch is None:
            self.exhausted = True
        return batch

    def prefetch_next(self):
        """在当前前向已提交后调用；关闭预取时不提前消费加载器。"""
        if not self.enabled or self.pending is not None or self.exhausted:
            return
        self.inflight = [entry for entry in self.inflight if not entry[0].query()]
        if len(self.inflight) >= 2:
            # 仅在 CPU 生产过快时做批次级背压，防止锁页内存无限增长。
            oldest = self.inflight.pop(0)
            oldest[0].synchronize()  # 等待期间仍持有整组 CPU 源张量。
        batch = self._next_cpu_batch()
        if batch is None:
            return
        samples, targets = batch
        # 正常 DataLoader 已提供锁页图像；只对未锁页输入做一次兼容转换。
        samples = samples if samples.is_pinned() else samples.pin_memory()
        sizes = torch.stack([target["orig_size"] for target in targets]).pin_memory()
        with torch.cuda.stream(self.stream):
            with self.section("h2d", gpu=True):
                gpu_samples = samples.to(self.device, non_blocking=True)
                gpu_sizes = sizes.to(self.device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        # 持有源张量直到 DMA 完成；代码不修改它们，也不手动复用其存储。
        self.inflight.append((ready, samples, sizes))
        self.pending = (gpu_samples, targets, gpu_sizes, ready)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.enabled:
            batch = self._next_cpu_batch()
            if batch is None:
                raise StopIteration
            samples, targets = batch
            with self.section("h2d", gpu=True):
                samples = samples.to(self.device)
                sizes = torch.stack([target["orig_size"] for target in targets]).to(self.device)
            return samples, targets, sizes
        self.prefetch_next()
        if self.pending is None:
            raise StopIteration
        samples, targets, sizes, ready = self.pending
        self.pending = None
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(ready)  # 只阻塞计算流，不在 CPU 上等待整张 GPU。
        samples.record_stream(stream)
        sizes.record_stream(stream)  # 防止拷贝流分配的显存被提前回收。
        return samples, targets, sizes

    def close(self):
        """正常退出通常已完成上传；异常或提前退出时仅等待尚未完成的拷贝。"""
        if self.inflight and not self.inflight[-1][0].query():
            self.inflight[-1][0].synchronize()
        self.inflight.clear()
        self.pending = None
        self.iterator = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
