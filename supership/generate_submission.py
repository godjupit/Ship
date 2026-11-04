"""Run RT-DETRv4 inference and write a COCO-format submission.

Supports the original one-shot mode and an organizer-controlled worker mode.
In worker mode, initialization happens before READY and test paths are received
over stdin only after the organizer starts the timer.
"""

import argparse
from functools import partial
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # Run from the RT-DETRv4 repository root.

import torch

from supership.cuda_graph_runner import CUDAGraphRunner


READY_MARKER = "THU_RS_READY"


def amp_enabled(device):
    """默认关闭 AMP；仅 CUDA 设备且 ENABLE_AMP=True/1/yes 时启用。"""
    switch = os.environ.get("ENABLE_AMP", "False").strip().lower()
    return torch.device(device).type == "cuda" and switch in ("true", "1", "yes")


def forward_model(model, samples, use_amp=False):
    """仅前向使用自动混合精度，后处理所需的类别和坐标恢复 FP32。"""
    if isinstance(model, CUDAGraphRunner):
        return model.forward(samples)
    if not use_amp:
        return model(samples)
    # 不需要 model.half() 或 GradScaler，autocast 自动选择算子精度。
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(samples)
    # FP32 后处理不会消除前向的数值差异，因此 AMP 必须重新评估 AP。
    return {**outputs, "pred_logits": outputs["pred_logits"].float(),
            "pred_boxes": outputs["pred_boxes"].float()}


def prepare_cuda_graph(model, device, batch_size):
    """按开关在 READY 前捕获；默认关闭，CPU 保持原始前向。"""
    enabled = os.environ.get("ENABLE_CUDA_GRAPH", "False").strip().lower()
    if enabled in ("true", "1", "yes") and torch.device(device).type == "cuda":
        eager_forward = partial(forward_model, model, use_amp=amp_enabled(device))
        return CUDAGraphRunner(eager_forward, device, batch_size)
    return model


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
    """按验证配置创建推理加载器，worker 数由 YAML 决定。"""
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
    return create("val_dataloader", cfg.global_cfg, batch_size=batch_size)


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
    use_amp = amp_enabled(device)
    results = []

    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.to(device)
            original_sizes = torch.stack(
                [target["orig_size"] for target in targets], dim=0
            ).to(device)
            outputs = forward_model(model, samples, use_amp)
            detections = postprocessor(outputs, original_sizes)

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
    model = prepare_cuda_graph(model, args.device, args.batch_size)

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
