"""离线导出前向 ONNX 并构建 TensorRT engine；不读取参考集图片。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from supership.generate_submission import build_runtime
from supership.inference_config import INFERENCE_CONFIG
from supership.tensorrt_runner import load_tensorrt, make_logger, model_fingerprint


class ForwardExport(torch.nn.Module):
    """只导出模型前向，保留仓库原有的 FP32 检测后处理。"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)
        return outputs["pred_logits"], outputs["pred_boxes"]


def main():
    """构建一次后缓存 engine；推理只加载，不在每轮重新编译。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dfine/dfine_hgnetv2_s_supership.yml")
    parser.add_argument("--weights", default="weights/baseline.pth")
    parser.add_argument("--output", default=INFERENCE_CONFIG.engine_path)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true", help="allow FP16 kernels inside TensorRT")
    args = parser.parse_args()
    if args.max_batch_size < 1:
        parser.error("--max-batch-size must be positive")
    trt = load_tensorrt()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = path.with_suffix(".onnx")
    cfg, model, _ = build_runtime(args.config, args.weights, args.device, backend="pytorch")
    fingerprint = model_fingerprint(cfg.yaml_cfg, args.weights)
    with torch.inference_mode():
        # batch>1 会导出 anchors 的动态 repeat；batch=1 时该 repeat 仍数学等价。
        # 若用单张样例跟踪，原模型的 Python 分支会把 anchors 固定成 batch=1。
        sample = torch.zeros((min(2, args.max_batch_size), 3, 640, 640), device=args.device)
        torch.onnx.export(
            ForwardExport(model).eval(), sample, str(onnx_path), opset_version=17,
            input_names=["images"], output_names=["pred_logits", "pred_boxes"],
            dynamic_axes={name: {0: "batch"} for name in ("images", "pred_logits", "pred_boxes")},
            do_constant_folding=True,
        )
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))
    del model, sample, cfg
    torch.cuda.empty_cache()
    print(f"Exported {onnx_path}; building engine on {torch.cuda.get_device_name(args.device)}", flush=True)
    logger = make_logger(trt)
    with torch.cuda.device(args.device), trt.Builder(logger) as builder:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        onnx_parser = trt.OnnxParser(network, logger)
        if not onnx_parser.parse_from_file(str(onnx_path.resolve())):
            raise RuntimeError("\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
        settings = builder.create_builder_config()
        settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024**3)
        settings.clear_flag(trt.BuilderFlag.TF32)
        settings.max_aux_streams = 0
        if args.fp16:
            settings.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape("images", (1, 3, 640, 640),
                          (args.max_batch_size, 3, 640, 640), (args.max_batch_size, 3, 640, 640))
        settings.add_optimization_profile(profile)
        engine = builder.build_serialized_network(network, settings)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed")
        data = bytes(engine)
    metadata = {
        "model_fingerprint": fingerprint, "engine_sha256": hashlib.sha256(data).hexdigest(),
        "tensorrt_version": trt.__version__, "gpu_name": torch.cuda.get_device_name(args.device),
        "max_batch_size": args.max_batch_size, "precision": "fp16" if args.fp16 else "fp32",
    }
    temporary = Path(str(path) + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    Path(str(path) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {path}; precision={metadata['precision']}, batch=1..{args.max_batch_size}", flush=True)


if __name__ == "__main__":
    main()
