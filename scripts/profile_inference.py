"""Standalone stage profiler; the competition worker remains unchanged."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from supership.generate_submission import build_runtime, create_loader, read_worker_request, READY_MARKER

class StageTimings:
    """Host wall times and CUDA stream times, without per-stage sync."""

    def __init__(self, device, enabled=None):
        # 默认关闭；环境变量会自动传给 benchmark 启动的 worker 子进程。
        if enabled is None:
            switch = os.environ.get("ENABLE_PROFILING", "False").strip().lower()
            enabled = switch in ("1", "true", "yes")
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self.wall = {}
        self.events = []
        self.started = time.perf_counter() if self.enabled else 0.0
        self.batches = 0
        self.images = 0
        self.h2d_bytes = 0

    @contextmanager
    def section(self, name, gpu=False):
        """GPU 用事件记录实际流耗时，CPU 用时钟；每个阶段结束时不强制同步。"""
        if not self.enabled:
            yield
            return
        if gpu and self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(self.device)
            start.record(stream)
            try:
                yield
            finally:
                end.record(stream)
                self.events.append((name + "_gpu_s", start, end))
        else:
            started = time.perf_counter()
            try:
                yield
            finally:
                self.add_wall(name, time.perf_counter() - started)

    @contextmanager
    def modules(self, model):
        """临时挂钩三个模块的输入和输出，测量时间但不改 forward 或张量。"""
        if not self.enabled:
            # 关闭时完全不注册挂钩，也不创建 CUDA Event。
            yield
            return
        handles = []

        def attach(name, module):
            active = []

            def before(module, inputs):
                # 在模块执行前记录起点；保存上下文以便结束时配对。
                section = self.section(name, gpu=True)
                section.__enter__()
                active.append(section)

            def after(module, inputs, output):
                # 只记录终点，不返回新输出，因此不会改变模型计算结果。
                if active:
                    active.pop().__exit__(None, None, None)

            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after, always_call=True))

        try:
            for name in ("backbone", "encoder", "decoder"):
                attach(name, getattr(model, name))
            yield
        finally:
            # 推理结束或报错都移除挂钩，避免重复运行时累积统计。
            for handle in handles:
                handle.remove()

    def add_wall(self, name, seconds):
        if self.enabled:
            key = name + "_wall_s"
            self.wall[key] = self.wall.get(key, 0.0) + seconds

    def report(self):
        if not self.enabled:
            return {"profiling_enabled": False}
        values = dict(self.wall)
        values["profiling_enabled"] = True
        # run_inference has already synchronized CUDA before writing output.
        for name, start, end in self.events:
            values[name] = values.get(name, 0.0) + start.elapsed_time(end) / 1000.0
        values.update(
            batches=self.batches, images=self.images, h2d_bytes=self.h2d_bytes,
            total_wall_s=time.perf_counter() - self.started,
        )
        def pad(text, width):
            used = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
            return text + " " * max(0, width - used)

        gpu = self.device.type == "cuda"
        suffix = "_gpu_s" if gpu else "_wall_s"
        rows = [
            ("创建加载器 / 读取元数据", "loader_create_wall_s", False),
            ("等待数据 / 解码与预处理", "loader_wait_wall_s", True),
            ("CPU → GPU 数据传输" if gpu else "数据设备转换", "h2d" + suffix, True),
            ("模型前向计算", "forward" + suffix, True),
            ("  Backbone 特征提取", "backbone" + suffix, True),
            ("  Encoder 特征融合", "encoder" + suffix, True),
            ("  Decoder 检测预测", "decoder" + suffix, True),
            ("模型后处理", "postprocess" + suffix, True),
            ("结果回传 CPU / 整理", "results_wall_s", True),
            ("JSON 写盘 / 刷盘", "write_wall_s", False),
        ]
        lines = [
            "", "========== 本轮分项计时 ==========",
            f"设备：{self.device}    图像：{self.images} 张    批次：{self.batches} 批",
            f"输入传输总量：{self.h2d_bytes / 1024**2:.2f} MiB",
            pad("阶段", 30) + "累计耗时（秒）    平均每批（毫秒）",
            "-" * 67,
        ]
        for label, key, per_batch in rows:
            seconds = values.get(key, 0.0)
            average = f"{seconds * 1000 / self.batches:.3f}" if per_batch and self.batches else "—"
            lines.append(pad(label, 30) + f"{seconds:>12.4f}    {average:>18}")
        lines += [
            "-" * 67,
            f"推理函数总耗时：{values['total_wall_s']:.4f} 秒",
            "说明：加载项统计主进程等待时间；结果整理包含同步等待。",
            "      CPU 与 GPU 阶段可能重叠，各项不能直接相加。",
            "      三个模块属于模型前向的子项，不要与前向总时间重复相加。",
            "      包含首批冷启动及事件挂钩开销，仅用于定位瓶颈。",
            "      比赛端到端时间见最后汇总，还包含 worker 退出等开销。",
            "=================================", "",
        ]
        print("\n".join(lines), file=sys.stderr, flush=True)
        return values


def run_inference(
    cfg,
    model,
    postprocessor,
    device,
    batch_size,
    image_dir,
    meta_file,
    output_file,
):
    # Loader creation is timed because it reads the evaluation metadata.
    timing = StageTimings(device)
    with timing.section("loader_create"):
        loader = create_loader(cfg, image_dir, meta_file, batch_size)
    results = []

    previous_batch_end = time.perf_counter() if timing.enabled else 0.0
    with torch.inference_mode(), timing.modules(model):
        for samples, targets in loader:
            if timing.enabled:
                timing.add_wall("loader_wait", time.perf_counter() - previous_batch_end)
            if timing.enabled:
                timing.batches += 1
                timing.images += len(targets)
                timing.h2d_bytes += samples.numel() * samples.element_size()
                timing.h2d_bytes += sum(t["orig_size"].numel() * t["orig_size"].element_size() for t in targets)
            # 与正式入口一致，直接在当前流上传本批数据。
            with timing.section("h2d", gpu=True):
                samples = samples.to(device)
                original_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            with timing.section("forward", gpu=True):
                outputs = model(samples)
            with timing.section("postprocess", gpu=True):
                detections = postprocessor(outputs, original_sizes)

            result_started = time.perf_counter() if timing.enabled else 0.0
            for target, detection in zip(targets, detections):
                image_id = target["image_id"].item()
                # Transfer all boxes once, avoiding one CUDA copy per query.
                boxes = detection["boxes"].tolist()
                scores = detection["scores"].tolist()
                labels = detection["labels"].tolist()
                for index, score in enumerate(scores):
                    x1, y1, x2, y2 = boxes[index]
                    width = round(x2 - x1, 2)
                    height = round(y2 - y1, 2)
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(labels[index]),
                            "bbox": [
                                round(x1, 2),
                                round(y1, 2),
                                width,
                                height,
                            ],
                            "score": round(score, 5),
                        }
                    )

            if timing.enabled:
                timing.add_wall("results", time.perf_counter() - result_started)
                previous_batch_end = time.perf_counter()

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

    with timing.section("write"):
        output_path = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output:
            json.dump(results, output, separators=(",", ":"), allow_nan=False)
            output.flush()
            os.fsync(output.fileno())

    values = timing.report()
    values.update(
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", False),
        batch_size=batch_size,
    )
    Path(output_file).with_suffix(".timing.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    return len(results)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", default="configs/dfine/dfine_hgnetv2_s_supership.yml")
    parser.add_argument("--weights", default="weights/baseline.pth")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--image-dir", default="data/SuperShip/reference/images")
    parser.add_argument("--meta-file", default="data/SuperShip/reference/reference_images_meta.json")
    parser.add_argument("--output", default="analysis/profile_submission.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    if not args.worker:
        command = [
            sys.executable, str(ROOT / "supership/benchmark_submission.py"),
            "--image-dir", args.image_dir, "--meta-file", args.meta_file,
            "--output-file", args.output, "--runs", str(args.runs), "--",
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--config", args.config, "--weights", args.weights,
            "--device", args.device, "--batch-size", str(args.batch_size),
        ]
        print(f"开始计时：{args.runs} 轮，batch_size={args.batch_size}，设备={args.device}", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        print(completed.stderr, end="", file=sys.stderr, flush=True)
        completed.check_returncode()
        summary = json.loads(completed.stdout.strip().splitlines()[-1])
        from supership.benchmark_submission import output_path_for_run
        runs = []
        report_lines = ["", "========== 端到端计时汇总 =========="]
        for index, seconds in enumerate(summary["runs_seconds"]):
            result_path = Path(output_path_for_run(args.output, index, args.runs))
            timing_path = result_path.with_suffix(".timing.json")
            run = json.loads(timing_path.read_text(encoding="utf-8"))
            run.update(end_to_end_s=seconds, output_file=str(result_path))
            runs.append(run)
            timing_path.unlink()
            report_lines.append(f"第 {index + 1} 轮：{seconds:.4f} 秒")
        report_lines.append(f"中位数：{summary['median_seconds']:.4f} 秒")
        result_dir = Path(args.output).parent
        report_lines.append(f"检测结果及分析文件：{result_dir}/")
        report = "\n".join(report_lines) + "\n"
        print(report, flush=True)
        (result_dir / "profile_report.txt").write_text(completed.stderr + report, encoding="utf-8")
        metrics = dict(
            device=args.device, config=args.config, weights=args.weights,
            image_dir=args.image_dir, meta_file=args.meta_file,
            median_end_to_end_s=summary["median_seconds"], runs=runs,
            note="分项可能重叠，不能相加；累计统计不是逐 batch 时间线。",
        )
        (result_dir / "profile_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return
    cfg, model, postprocessor = build_runtime(args.config, args.weights, args.device)
    print(READY_MARKER, flush=True)
    request = read_worker_request()
    count = run_inference(
        cfg, model, postprocessor, args.device, args.batch_size,
        request["image_dir"], request["meta_file"], request["output_file"],
    )
    print(f"检测完成：共输出 {count} 条检测结果。", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
