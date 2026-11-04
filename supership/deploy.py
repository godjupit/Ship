"""Export raw predictions to ONNX and build fixed-batch TensorRT engines."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from supership.generate_submission import build_runtime


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Predictions(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)
        return outputs["pred_logits"], outputs["pred_boxes"]


def export(args):
    import onnx
    import onnxruntime as ort
    import numpy as np
    from PIL import Image
    from torchvision.transforms import functional as F
    from scipy.optimize import linear_sum_assignment
    from scipy.special import expit

    _, model, _ = build_runtime(args.config, args.weights, args.device)
    model = Predictions(model.deploy().to(args.device)).eval()
    # Real training samples avoid near-tied proposal rankings on random noise.
    training = json.loads((ROOT / "data/SuperShip/train/annotations/instances_train.json").read_text())
    images = training["images"][:args.batch_size]
    tensors = []
    for item in images:
        with Image.open(ROOT / "data/SuperShip/train/images" / item["file_name"]) as source:
            tensors.append(F.pil_to_tensor(F.resize(source.convert("RGB"), [640, 640])).float() / 255.)
    if len(tensors) != args.batch_size:
        raise ValueError("Not enough training images for export verification")
    sample = torch.stack(tensors).to(args.device)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = model(sample)
        torch.onnx.export(model, sample, str(target), opset_version=17,
                          input_names=["images"], output_names=["pred_logits", "pred_boxes"],
                          do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(target))
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(target), sess_options=session_options,
                                   providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": sample.cpu().numpy()})
    errors = {"matched_box_max_abs": 0., "matched_probability_max_abs": 0., "checked_predictions": 0}
    references = [value.cpu().numpy() for value in expected]
    # Query order may differ after TopK; compare sets, not query array positions.
    for ref_logits, ref_boxes, logits, boxes in zip(*references, *actual):
        ref_probs, probs = expit(ref_logits), expit(logits)
        cost = np.abs(ref_boxes[:, None] - boxes[None]).sum(-1)
        cost += np.abs(ref_probs[:, None] - probs[None]).sum(-1)
        left, right = linear_sum_assignment(cost)
        keep = np.maximum(ref_probs[left].max(-1), probs[right].max(-1)) >= 0.3
        left, right = left[keep], right[keep]
        if not len(left):
            continue
        errors["checked_predictions"] += len(left)
        errors["matched_box_max_abs"] = max(errors["matched_box_max_abs"], float(np.abs(ref_boxes[left] - boxes[right]).max()))
        errors["matched_probability_max_abs"] = max(errors["matched_probability_max_abs"], float(np.abs(ref_probs[left] - probs[right]).max()))
    if not errors["checked_predictions"] or errors["matched_box_max_abs"] > 0.02 or errors["matched_probability_max_abs"] > 0.03:
        raise RuntimeError(f"ONNX prediction-set verification failed: {errors}")
    manifest = dict(config=args.config, weights=args.weights, weights_sha256=digest(args.weights),
                    batch_size=args.batch_size, onnx_sha256=digest(target), onnx_errors=errors,
                    verification_split="train", verification_images=[item["file_name"] for item in images])
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(manifest, indent=2))
    print(f"ONNX checked: {target}; max absolute errors: {errors}")


def calibrator(trt, batch_size, sample_count, seed, image_dir, annotation, cache):
    from PIL import Image
    from torchvision.transforms import functional as F

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            images = json.loads(Path(annotation).read_text())["images"]
            random.Random(seed).shuffle(images)
            self.images = images[:sample_count]
            if len(self.images) < batch_size:
                raise ValueError("Calibration needs at least one complete training batch")
            self.position = 0
            self.device_buffer = torch.empty(batch_size, 3, 640, 640, device="cuda")
            self.used_files = []

        def get_batch_size(self):
            return batch_size

        def get_batch(self, names):
            if names != ["images"]:
                raise ValueError(f"Unexpected calibration bindings: {names}")
            if self.position + batch_size > len(self.images):
                return None
            tensors = []
            for item in self.images[self.position:self.position + batch_size]:
                with Image.open(Path(image_dir) / item["file_name"]) as source:
                    image = F.resize(source.convert("RGB"), [640, 640])
                    tensors.append(F.pil_to_tensor(image).float() / 255.)
                self.used_files.append(item["file_name"])
            self.device_buffer.copy_(torch.stack(tensors))
            torch.cuda.synchronize()
            self.position += batch_size
            return [self.device_buffer.data_ptr()]

        def read_calibration_cache(self):
            # Recalibrate instead of silently accepting a cache from other weights/data.
            return None

        def write_calibration_cache(self, data):
            Path(cache).write_bytes(data)

    return EntropyCalibrator()


def quantize(args):
    """Explicit symmetric INT8 Q/DQ, restricted to backbone convolutions."""
    import onnx
    import numpy as np
    from PIL import Image
    from torchvision.transforms import functional as F
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          QuantFormat, QuantType, quantize_static)

    source_manifest = json.loads(Path(args.onnx + ".json").read_text())
    if source_manifest.get("onnx_sha256") != digest(args.onnx):
        raise ValueError("Source ONNX checksum mismatch")
    model = onnx.load(args.onnx)
    batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
    if batch_size < 1 or args.calib_samples < batch_size:
        raise ValueError("Calibration requires at least one complete static batch")
    nodes = [node.name for node in model.graph.node if node.op_type == "Conv"
             and (args.int8_scope == "all_conv" or "/backbone/" in node.name)]
    if not nodes:
        raise ValueError("No convolution nodes selected for quantization")
    images = json.loads((ROOT / "data/SuperShip/train/annotations/instances_train.json").read_text())["images"]
    random.Random(args.seed).shuffle(images)
    images = images[:args.calib_samples]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.position = 0
            self.files = []

        def get_next(self):
            if self.position + batch_size > len(images):
                return None
            tensors = []
            for item in images[self.position:self.position + batch_size]:
                with Image.open(ROOT / "data/SuperShip/train/images" / item["file_name"]) as source:
                    tensors.append((F.pil_to_tensor(F.resize(source.convert("RGB"), [640, 640])).float() / 255.).numpy())
                self.files.append(item["file_name"])
            self.position += batch_size
            print(f"INT8 calibration: {self.position}/{len(images)} training images", flush=True)
            return {"images": np.stack(tensors)}

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    reader = Reader()
    quantize_static(args.onnx, str(target), reader, quant_format=QuantFormat.QDQ,
                    nodes_to_quantize=nodes, op_types_to_quantize=["Conv"], per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                                   "QuantizeBias": False})
    onnx.checker.check_model(onnx.load(target))
    manifest = source_manifest
    manifest["source_fp32_verification"] = manifest.pop("onnx_errors", {})
    manifest.update(onnx_sha256=digest(target), quantization="symmetric INT8 QDQ",
                    int8_scope=args.int8_scope, quantized_conv_nodes=len(nodes),
                    calibration_split="train", calibration_files=reader.files, seed=args.seed)
    Path(str(target) + ".json").write_text(json.dumps(manifest, indent=2))
    print(f"INT8 QDQ ONNX saved: {target}; convolutions: {len(nodes)}", flush=True)


def build(args):
    import tensorrt as trt
    import onnx

    explicit_int8 = any(node.op_type == "QuantizeLinear" for node in onnx.load(args.onnx).graph.node)
    if args.precision == "int8" and args.int8_mode == "qdq" and not explicit_int8:
        output_onnx = args.output + ".qdq.onnx"
        quantize(argparse.Namespace(onnx=args.onnx, output=output_onnx,
                                   calib_samples=args.calib_samples, seed=args.seed,
                                   int8_scope=args.int8_scope))
        args.onnx = output_onnx
        explicit_int8 = True
    source_manifest = Path(args.onnx + ".json")
    if not source_manifest.exists():
        raise ValueError("Missing ONNX manifest; export with supership/deploy.py first")
    manifest = json.loads(source_manifest.read_text())
    if manifest.get("onnx_sha256") != digest(args.onnx):
        raise ValueError("ONNX checksum mismatch; export again before building")
    torch.cuda.set_device(args.device)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(args.onnx).read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    shape = tuple(network.get_input(0).shape)
    if len(shape) != 4 or shape[1:] != (3, 640, 640) or shape[0] < 1:
        raise ValueError(f"Expected static N×3×640×640 input, got {shape}")
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib << 30)
    config.builder_optimization_level = args.optimization_level
    if args.precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    calibration = None
    if args.precision == "int8" and not explicit_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # Calibration exclusively uses the released training split.
        calibration = calibrator(trt, shape[0], args.calib_samples, args.seed,
                                 ROOT / "data/SuperShip/train/images",
                                 ROOT / "data/SuperShip/train/annotations/instances_train.json",
                                 str(target) + ".calibration.cache")
        config.int8_calibrator = calibration
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    target.write_bytes(serialized)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Built engine cannot be deserialized")
    inspection = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    Path(str(target) + ".layers.json").write_text(inspection)
    manifest.update(tensorrt=trt.__version__, precision=args.precision,
                    gpu=torch.cuda.get_device_name(), engine_sha256=digest(target),
                    onnx_sha256=digest(args.onnx), batch_size=shape[0],
                    builder_optimization_level=args.optimization_level, workspace_gib=args.workspace_gib,
                    calibration_split="train" if calibration else manifest.get("calibration_split"),
                    calibration_files=calibration.used_files if calibration else manifest.get("calibration_files", []), seed=args.seed,
                    inspector_reports_int8="int8" in inspection.lower())
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(manifest, indent=2))
    print(f"TensorRT {args.precision} engine saved: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--config", default="configs/dfine/dfine_hgnetv2_s_supership.yml")
    exp.add_argument("--weights", default="weights/baseline.pth")
    exp.add_argument("--batch-size", type=int, default=8)
    exp.add_argument("--device", default="cuda:0")
    exp.add_argument("--output", default="output/deploy/model.onnx")
    eng = sub.add_parser("build")
    eng.add_argument("--onnx", default="output/deploy/model.onnx")
    eng.add_argument("--output", required=True)
    eng.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    eng.add_argument("--device", default="cuda:0")
    eng.add_argument("--workspace-gib", type=int, default=2)
    eng.add_argument("--optimization-level", type=int, choices=range(6), default=3)
    eng.add_argument("--calib-samples", type=int, default=128)
    eng.add_argument("--seed", type=int, default=42)
    eng.add_argument("--int8-mode", choices=["qdq", "legacy"], default="qdq")
    eng.add_argument("--int8-scope", choices=["backbone", "all_conv"], default="backbone")
    quant = sub.add_parser("quantize")
    quant.add_argument("--onnx", default="output/deploy/model.onnx")
    quant.add_argument("--output", default="output/deploy/model.int8.onnx")
    quant.add_argument("--calib-samples", type=int, default=128)
    quant.add_argument("--seed", type=int, default=42)
    quant.add_argument("--int8-scope", choices=["backbone", "all_conv"], default="backbone")
    args = parser.parse_args()
    if args.command == "export" and args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.command == "build" and (args.calib_samples < 1 or args.workspace_gib < 1):
        parser.error("calib-samples and workspace-gib must be positive")
    {"export": export, "build": build, "quantize": quantize}[args.command](args)


if __name__ == "__main__":
    main()
