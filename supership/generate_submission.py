"""Run RT-DETRv4 inference and write a COCO-format submission.

Supports the original one-shot mode and an organizer-controlled worker mode.
In worker mode, initialization happens before READY and test paths are received
over stdin only after the organizer starts the timer.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # Run from the RT-DETRv4 repository root.

import torch


READY_MARKER = "THU_RS_READY"


def build_runtime(config_path, weights_path, device):
    from engine.core import YAMLConfig
    if os.environ.get("INFERENCE_IMPORTS", "0") != "1":
        from engine.data.dataset.coco_dataset import CocoDetection  # noqa: F401
        from engine.solver.det_engine import evaluate  # noqa: F401

    cfg = YAMLConfig(config_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    state = torch.load(weights_path, map_location="cpu")
    for component, values in state.get("model_overrides", {}).items():
        cfg.yaml_cfg.setdefault(component, {}).update(values)
    model = cfg.model.to(device)
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


def create_loader(cfg, image_dir, meta_file, batch_size, options=None):
    from engine.data.dataset.coco_dataset import CocoDetection  # noqa: F401
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
    if options is not None:
        loader_cfg = cfg.yaml_cfg["val_dataloader"]
        if options.num_workers is not None:
            loader_cfg["num_workers"] = options.num_workers
        loader_cfg["pin_memory"] = options.pin_memory
        if loader_cfg.get("num_workers", 0) > 0:
            loader_cfg["prefetch_factor"] = options.prefetch_factor
        else:
            loader_cfg.pop("prefetch_factor", None)
    return create("val_dataloader", cfg.global_cfg, batch_size=batch_size)


def model_forward(model, samples, precision):
    if getattr(model, "is_runtime", False):
        return model(samples)
    with torch.autocast(device_type=samples.device.type, dtype=torch.float16,
                        enabled=precision == "fp16"):
        outputs = model(samples)
    if precision == "fp16":
        # Keep coordinate scaling and score processing in float32.
        outputs["pred_boxes"] = outputs["pred_boxes"].float()
        outputs["pred_logits"] = outputs["pred_logits"].float()
    return outputs


def run_inference(
    cfg,
    model,
    postprocessor,
    device,
    batch_size,
    image_dir,
    meta_file,
    output_file,
    options=None,
):
    # Loader creation is timed because it reads the evaluation metadata.
    loader = create_loader(cfg, image_dir, meta_file, batch_size, options)
    precision = options.precision if options else "fp32"
    non_blocking = options.non_blocking if options else False
    results = []

    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.to(device, non_blocking=non_blocking)
            original_sizes = torch.stack(
                [target["orig_size"] for target in targets], dim=0
            ).to(device, non_blocking=non_blocking)
            outputs = model_forward(model, samples, precision)
            detections = postprocessor(outputs, original_sizes)

            if options and options.batch_copy:
                boxes_batch = torch.stack([d["boxes"] for d in detections]).cpu().tolist()
                scores_batch = torch.stack([d["scores"] for d in detections]).cpu().tolist()
                labels_batch = torch.stack([d["labels"] for d in detections]).cpu().tolist()
                detections = [dict(boxes=b, scores=s, labels=l)
                              for b, s, l in zip(boxes_batch, scores_batch, labels_batch)]
            for target, detection in zip(targets, detections):
                image_id = target["image_id"].item()
                # Transfer each tensor once; batch-copy mode already provides lists.
                boxes = detection["boxes"]
                scores = detection["scores"]
                labels = detection["labels"]
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.tolist()
                if isinstance(scores, torch.Tensor):
                    scores = scores.tolist()
                if isinstance(labels, torch.Tensor):
                    labels = labels.tolist()
                for index, score in enumerate(scores):
                    x1, y1, x2, y2 = boxes[index]
                    width = round(x2 - x1, 2)
                    height = round(y2 - y1, 2)
                    # Preserve non-positive boxes for the upstream unmatched-detection rule.
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(labels[index]),
                            "bbox": [round(x1, 2), round(y1, 2), width, height],
                            "score": round(score, 5),
                        }
                    )

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

    output_path = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output:
        if options and options.buffered_json:
            output.write(json.dumps(results, separators=(",", ":"), allow_nan=False))
        else:
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
    parser.add_argument("--batch-copy", action="store_true", default=os.environ.get("BATCH_COPY", "0") == "1")
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "0")))
    parser.add_argument("--precision", choices=["fp32", "fp16"], default=os.environ.get("PRECISION", "fp32"))
    parser.add_argument("--num-workers", type=int, default=int(os.environ["NUM_WORKERS"]) if "NUM_WORKERS" in os.environ else None)
    parser.add_argument("--prefetch-factor", type=int, default=int(os.environ.get("PREFETCH_FACTOR", "2")))
    parser.add_argument("--pin-memory", action="store_true", default=os.environ.get("PIN_MEMORY", "0") == "1")
    parser.add_argument("--non-blocking", action="store_true", default=os.environ.get("NON_BLOCKING", "0") == "1")
    parser.add_argument("--buffered-json", action="store_true", default=os.environ.get("BUFFERED_JSON", "0") == "1")
    parser.add_argument("--backend", choices=["torch", "cuda_graph", "tensorrt"], default=os.environ.get("BACKEND", "torch"))
    parser.add_argument("--engine", default=os.environ.get("ENGINE_PATH"))
    parser.add_argument("--cuda-graph", action="store_true", default=os.environ.get("CUDA_GRAPH", "0") == "1")
    parser.add_argument(
        "--worker",
        action="store_true",
        help="initialize, print READY, then receive test paths as one JSON line",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须为正数")
    if args.warmup < 0 or args.prefetch_factor < 1 or (args.num_workers is not None and args.num_workers < 0):
        parser.error("warmup 和 num-workers 必须非负，prefetch-factor 必须为正数")
    if args.precision == "fp16" and not args.device.startswith("cuda"):
        parser.error("FP16 实验仅支持 CUDA 设备")
    if args.backend == "tensorrt" and not args.engine:
        parser.error("TensorRT requires --engine or ENGINE_PATH")
    return args


def main():
    args = parse_args()
    cfg, model, postprocessor = build_runtime(args.config, args.weights, args.device)
    if args.backend == "tensorrt":
        from supership.backends import TensorRTRuntime
        model = TensorRTRuntime(args.engine, args.batch_size, args.device, args.weights)
        model.is_runtime = True
    if args.backend == "cuda_graph" or args.cuda_graph:
        from supership.backends import CUDAGraphRuntime
        model = CUDAGraphRuntime(model, args.batch_size, args.device, args.precision)
        model.is_runtime = True
    if args.warmup:
        # Synthetic inputs only: evaluation paths are received after READY.
        with torch.inference_mode():
            dummy = torch.rand(args.batch_size, 3, 640, 640, device=args.device)
            sizes = torch.full((args.batch_size, 2), 640, device=args.device)
            for _ in range(args.warmup):
                postprocessor(model_forward(model, dummy, args.precision), sizes)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        del dummy, sizes

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
            options=args,
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
            options=args,
        )

    print(f"wrote {count} detections", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
