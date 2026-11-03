"""Standalone stage profiler; the competition worker remains unchanged."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from supership.generate_submission import build_runtime, create_loader, read_worker_request, READY_MARKER

class StageTimings:
    """Host wall times and CUDA stream times, without per-stage sync."""

    def __init__(self, device):
        self.enabled = True
        self.device = torch.device(device)
        self.wall = {}
        self.events = []
        self.started = time.perf_counter()
        self.batches = 0
        self.images = 0
        self.h2d_bytes = 0

    @contextmanager
    def section(self, name, gpu=False):
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

    def add_wall(self, name, seconds):
        if self.enabled:
            key = name + "_wall_s"
            self.wall[key] = self.wall.get(key, 0.0) + seconds

    def report(self):
        if not self.enabled:
            return
        values = dict(self.wall)
        # run_inference has already synchronized CUDA before writing output.
        for name, start, end in self.events:
            values[name] = values.get(name, 0.0) + start.elapsed_time(end) / 1000.0
        values.update(
            batches=self.batches, images=self.images, h2d_bytes=self.h2d_bytes,
            total_wall_s=time.perf_counter() - self.started,
        )
        print("PROFILE " + json.dumps(values, sort_keys=True), file=sys.stderr, flush=True)


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
    with torch.inference_mode():
        for samples, targets in loader:
            timing.add_wall("loader_wait", time.perf_counter() - previous_batch_end)
            if timing.enabled:
                timing.batches += 1
                timing.images += len(targets)
                timing.h2d_bytes += samples.numel() * samples.element_size()
                timing.h2d_bytes += sum(t["orig_size"].numel() * t["orig_size"].element_size() for t in targets)
            with timing.section("h2d", gpu=True):
                samples = samples.to(device)
                original_sizes = torch.stack(
                    [target["orig_size"] for target in targets], dim=0
                ).to(device)
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

            timing.add_wall("results", time.perf_counter() - result_started)
            previous_batch_end = time.perf_counter() if timing.enabled else 0.0

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

    with timing.section("write"):
        output_path = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output:
            json.dump(results, output, separators=(",", ":"), allow_nan=False)
            output.flush()
            os.fsync(output.fileno())

    timing.report()

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
    parser.add_argument("--output", default="output/profile_submission.json")
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
        subprocess.run(command, check=True)
        return
    cfg, model, postprocessor = build_runtime(args.config, args.weights, args.device)
    print(READY_MARKER, flush=True)
    request = read_worker_request()
    count = run_inference(
        cfg, model, postprocessor, args.device, args.batch_size,
        request["image_dir"], request["meta_file"], request["output_file"],
    )
    print(f"wrote {count} detections", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
