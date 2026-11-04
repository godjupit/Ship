"""Run RT-DETRv4 inference and write a COCO-format submission.

Supports the original one-shot mode and an organizer-controlled worker mode.
In worker mode, initialization happens before READY and test paths are received
over stdin only after the organizer starts the timer.
"""

import argparse
from contextlib import nullcontext
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # Run from the RT-DETRv4 repository root.

import torch


READY_MARKER = "THU_RS_READY"


class DevicePrefetcher:
    """用独立 CUDA 流预取下一批；CPU 推理使用普通设备转换。"""

    def __init__(self, loader, device, section=None):
        self.device = torch.device(device)
        self.section = section or (lambda *args, **kwargs: nullcontext())
        self.stream = (torch.cuda.Stream(device=self.device)
                       if self.device.type == "cuda" else None)
        self.pending = None
        self.inflight = []
        with self.section("loader_wait"):
            self.iterator = iter(loader)
        self.preload_next()

    def preload_next(self):
        """取出下一批并提交上传；在当前批计算提交后调用。"""
        self.inflight = [entry for entry in self.inflight if not entry[0].query()]
        with self.section("loader_wait"):
            batch = next(self.iterator, None)
        if batch is None:
            self.pending = None
            return
        samples, targets = batch
        sizes = torch.stack([target["orig_size"] for target in targets])
        if self.stream is None:
            with self.section("h2d", gpu=True):
                self.pending = (samples.to(self.device), targets, sizes.to(self.device), None)
            return
        # 图片和尺寸均使用锁页内存，并保留 CPU 引用直到 DMA 上传完成。
        if not samples.is_pinned():
            samples = samples.pin_memory()
        sizes = sizes.pin_memory()
        with torch.cuda.stream(self.stream):
            with self.section("h2d", gpu=True):
                gpu_samples = samples.to(self.device, non_blocking=True)
                gpu_sizes = sizes.to(self.device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        self.inflight.append((ready, samples, sizes))
        self.pending = (gpu_samples, targets, gpu_sizes, ready)

    def __iter__(self):
        return self

    def __next__(self):
        """交出已预取的数据，让计算流等待对应批次的上传事件。"""
        if self.pending is None:
            raise StopIteration
        samples, targets, sizes, ready = self.pending
        self.pending = None
        if ready is not None:
            stream = torch.cuda.current_stream(self.device)
            # 只让 GPU 计算流等待；record_stream 防止显存被分配器提前复用。
            stream.wait_event(ready)
            samples.record_stream(stream)
            sizes.record_stream(stream)
        return samples, targets, sizes

    def close(self):
        """退出时等上传结束，保证提前退出或异常时 CPU 数据也能安全释放。"""
        if self.stream is not None:
            self.stream.synchronize()
        self.inflight.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def batch_detections_to_cpu(detections):
    """按字段合并整批结果后回传，保留标签整数类型和每张图的检测数量。"""
    if not detections:
        return []
    counts = [detection["scores"].shape[0] for detection in detections]
    # 每个字段整批只回传一次，避免逐张图片触发小额 GPU→CPU 传输。
    columns = {
        key: torch.cat([detection[key] for detection in detections], dim=0).cpu().tolist()
        for key in ("boxes", "scores", "labels")
    }
    result = []
    offset = 0
    for count in counts:
        result.append({key: values[offset:offset + count] for key, values in columns.items()})
        offset += count
    return result


def build_runtime(config_path, weights_path, device):
    from engine.core import YAMLConfig
    from engine.data.dataset.coco_dataset import CocoDetection  # noqa: F401
    from engine.solver.det_engine import evaluate  # noqa: F401

    cfg = YAMLConfig(config_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = cfg.model.to(device)
    state = torch.load(weights_path, map_location="cpu")
    state_dict = (
        state["ema"]["module"]
        if "ema" in state and "module" in state.get("ema", {})
        else state["model"]
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    postprocessor = cfg.postprocessor.to(device)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

    return cfg, model, postprocessor


def create_loader(cfg, image_dir, meta_file, batch_size):
    from engine.core.workspace import create

    dataset_cfg = cfg.yaml_cfg["val_dataloader"]["dataset"]
    dataset_cfg["img_folder"] = image_dir
    dataset_cfg["ann_file"] = meta_file
    dataset_cfg["transforms"] = {
        "type": "Compose",
        "ops": [
            {"type": "Resize", "size": [640, 640]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        ],
    }
    cfg.yaml_cfg["val_dataloader"].pop("total_batch_size", None)
    return create("val_dataloader", cfg.global_cfg, batch_size=batch_size,pin_memory=True)


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
    loader = create_loader(cfg, image_dir, meta_file, batch_size)
    results = []

    with torch.inference_mode(), DevicePrefetcher(loader, device) as prefetcher:
        for samples, targets, original_sizes in prefetcher:
            outputs = model(samples)
            detections = postprocessor(outputs, original_sizes)
            # 在结果回传阻塞 CPU 前上传下一批，争取与当前批 GPU 计算重叠。
            prefetcher.preload_next()

            cpu_detections = batch_detections_to_cpu(detections)
            for target, detection in zip(targets, cpu_detections):
                image_id = target["image_id"].item()
                boxes = detection["boxes"]
                scores = detection["scores"]
                labels = detection["labels"]
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

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

    output_path = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(results, output, separators=(",", ":"), allow_nan=False)
        output.flush()
        os.fsync(output.fileno())

    return len(results)


def read_worker_request():
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("stdin closed before an inference request was received")
    request = json.loads(line)
    required = {"image_dir", "meta_file", "output_file"}
    missing = required.difference(request)
    if missing:
        raise ValueError(f"worker request is missing: {sorted(missing)}")
    return request


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("--img-folder")
    parser.add_argument("--ann-file")
    parser.add_argument("-o", "--output", default="submission.json")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--worker",
        action="store_true",
        help="initialize, print READY, then receive test paths as one JSON line",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg, model, postprocessor = build_runtime(args.config, args.weights, args.device)

    if args.worker:
        print(READY_MARKER, flush=True)
        request = read_worker_request()
        count = run_inference(
            cfg,
            model,
            postprocessor,
            args.device,
            args.batch_size,
            request["image_dir"],
            request["meta_file"],
            request["output_file"],
        )
    else:
        if not args.img_folder or not args.ann_file:
            raise SystemExit("--img-folder and --ann-file are required without --worker")
        count = run_inference(
            cfg,
            model,
            postprocessor,
            args.device,
            args.batch_size,
            args.img_folder,
            args.ann_file,
            args.output,
        )

    print(f"wrote {count} detections", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
