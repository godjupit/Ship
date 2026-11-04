"""Split DataLoader startup and batch waits while reusing production inference.

Run from the repository root after sourcing activate_env.sh:
    python scripts/experiments/profile_dataloader.py --runs 3

wall_ms includes host blocking; main_thread_cpu_ms excludes blocking and worker
CPU usage. CUDA stream readiness is only a boundary snapshot. For exact overlap,
capture this script with Nsight Systems (--trace=cuda,nvtx); dataloader/* ranges
mark the measured calls. Never add synchronize() around individual next() calls.
The uninstrumented warmup and every measured run create fresh workers. Model
initialization and process exit are excluded from total_seconds.
"""

import argparse
import hashlib
import json
import multiprocessing
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from supership import generate_submission as inference
from supership.inference_config import INFERENCE_CONFIG


class LoaderProfile:
    """Measure host waits without synchronizing CUDA or preloading batches."""

    def __init__(self, device):
        self.device = device
        self.cuda = str(device).startswith("cuda")
        self.records = []
        self.settings = {}

    def measure(self, name, operation):
        # query() is a nonblocking snapshot, not a CUDA synchronization. NVTX
        # ranges allow the exact GPU overlap to be checked in a separate trace.
        ready_before = torch.cuda.current_stream(self.device).query() if self.cuda else None
        if self.cuda:
            torch.cuda.nvtx.range_push(f"dataloader/{name}")
        cpu_start = time.thread_time_ns()
        start = time.perf_counter_ns()
        exhausted = False
        try:
            return operation()
        except StopIteration:
            exhausted = True
            raise
        finally:
            wall_ms = (time.perf_counter_ns() - start) / 1e6
            cpu_ms = (time.thread_time_ns() - cpu_start) / 1e6
            if self.cuda:
                torch.cuda.nvtx.range_pop()
            ready_after = torch.cuda.current_stream(self.device).query() if self.cuda else None
            self.records.append({
                "stage": "exhaustion" if exhausted else name,
                "wall_ms": wall_ms,
                "main_thread_cpu_ms": cpu_ms,
                "cuda_stream_ready_before": ready_before,
                "cuda_stream_ready_after": ready_after,
            })

    def wrap_factory(self, factory):
        def create_loader(*args, **kwargs):
            loader = self.measure("create_loader", lambda: factory(*args, **kwargs))
            self.settings = {
                "num_workers": loader.num_workers,
                "batch_size": loader.batch_size,
                "prefetch_factor": loader.prefetch_factor,
                "persistent_workers": loader.persistent_workers,
                "pin_memory": loader.pin_memory,
                "batches": len(loader),
                "images": len(loader.dataset),
                "multiprocessing_start_method": multiprocessing.get_start_method(),
            }
            return TimedLoader(loader, self)

        return create_loader


class TimedLoader:
    def __init__(self, loader, profile):
        self.loader = loader
        self.profile = profile

    def __iter__(self):
        iterator = self.profile.measure("iter_loader", lambda: iter(self.loader))
        return TimedIterator(iterator, self.profile)


class TimedIterator:
    def __init__(self, iterator, profile):
        self.iterator = iterator
        self.profile = profile
        self.batch_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_index += 1
        return self.profile.measure(f"next_{self.batch_index:02d}", lambda: next(self.iterator))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dfine/dfine_hgnetv2_s_supership.yml")
    parser.add_argument("--weights", default="weights/baseline.pth")
    parser.add_argument("--image-dir", default="data/SuperShip/reference/images")
    parser.add_argument("--meta-file", default="data/SuperShip/reference/reference_images_meta.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", default=f"analysis/dataloader/{time.strftime('%Y%m%d-%H%M%S')}")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    cfg, model, postprocessor = inference.build_runtime(args.config, args.weights, args.device)

    def run(output):
        start = time.perf_counter()
        count = inference.run_inference(
            cfg, model, postprocessor, args.device, args.batch_size,
            args.image_dir, args.meta_file, str(output),
        )
        return {"total_seconds": time.perf_counter() - start, "detections": count}

    # Warm the real pipeline once; its uninstrumented output is the equality
    # reference. Every subsequent run still creates a fresh loader and workers.
    baseline_path = directory / "warmup.json"
    baseline = run(baseline_path)
    baseline_digest = digest(baseline_path)
    reports = []
    factory = inference.create_loader
    for index in range(args.runs):
        profile = LoaderProfile(args.device)
        output = directory / f"run{index + 1}.json"
        with patch.object(inference, "create_loader", profile.wrap_factory(factory)):
            report = run(output)
        report.update({
            "run": index + 1,
            "settings": profile.settings,
            "records": profile.records,
            "output_matches_warmup": digest(output) == baseline_digest,
        })
        reports.append(report)
        print(json.dumps(report), flush=True)

    stages = [record["stage"] for record in reports[0]["records"]]
    medians = {
        stage: statistics.median(
            next(record["wall_ms"] for record in report["records"] if record["stage"] == stage)
            for report in reports
        )
        for stage in stages
    }
    totals = [{
        "startup_ms": sum(r["wall_ms"] for r in report["records"]
                          if r["stage"] in ("iter_loader", "next_01")),
        "later_next_ms": sum(r["wall_ms"] for r in report["records"]
                             if r["stage"].startswith("next_") and r["stage"] != "next_01"),
        "loader_wait_ms": sum(r["wall_ms"] for r in report["records"]
                              if r["stage"] != "create_loader"),
    } for report in reports]
    summary = {
        "torch_version": torch.__version__,
        "backend": INFERENCE_CONFIG.backend,
        "device": torch.cuda.get_device_name(args.device) if profile.cuda else args.device,
        "warmup": baseline,
        "median_total_seconds": statistics.median(r["total_seconds"] for r in reports),
        "stage_wall_median_ms": medians,
        "aggregate_median_ms": {key: statistics.median(r[key] for r in totals)
                                for key in totals[0]},
        "all_outputs_match": all(r["output_matches_warmup"] for r in reports),
        "runs": reports,
    }
    (directory / "timings.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}), flush=True)
    if not summary["all_outputs_match"]:
        raise SystemExit("Profile output differs from uninstrumented warmup")


if __name__ == "__main__":
    main()
