"""Build a static FP16 engine using checkpoint weights and synthetic inputs only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supership.generate_submission import build_runtime


class Predictions(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)
        return outputs["pred_logits"], outputs["pred_boxes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dfine/dfine_hgnetv2_s_supership.yml")
    parser.add_argument("--weights", default="weights/baseline.pth")
    parser.add_argument("--output", default="weights/supership-544.engine")
    parser.add_argument("--side", type=int, default=544)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not 320 <= args.side <= 1280 or args.side % 32 or args.batch_size < 1:
        parser.error("side must be a multiple of 32 in [320, 1280]; batch must be positive")

    import onnx
    import tensorrt as trt

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _, model, _ = build_runtime(args.config, args.weights, "cuda:0")
    encoder, decoder = model.encoder, model.decoder
    encoder.eval_spatial_size = decoder.eval_spatial_size = (args.side, args.side)
    for index in encoder.use_encoder_idx:
        stride = encoder.feat_strides[index]
        position = encoder.build_2d_sincos_position_embedding(
            args.side // stride, args.side // stride,
            encoder.hidden_dim, encoder.pe_temperature, device="cuda:0",
        )
        setattr(encoder, f"pos_embed{index}", position)
    decoder.anchors, decoder.valid_mask = decoder._generate_anchors(device="cuda:0")
    model.deploy().eval()
    shape = (args.batch_size, 3, args.side, args.side)
    with torch.inference_mode():
        sample = torch.zeros(shape, device="cuda:0")
        torch.onnx.export(
            Predictions(model), sample, str(output.with_suffix(".onnx")),
            opset_version=17, input_names=["images"],
            output_names=["pred_logits", "pred_boxes"], do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(output.with_suffix(".onnx")))
    del model, encoder, decoder, sample
    torch.cuda.empty_cache()
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    with trt.Builder(logger) as builder:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        onnx_parser = trt.OnnxParser(network, logger)
        if not onnx_parser.parse_from_file(str(output.with_suffix(".onnx").resolve())):
            raise RuntimeError("\n".join(str(onnx_parser.get_error(i))
                                         for i in range(onnx_parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024**3)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_flag(trt.BuilderFlag.FP16)
        config.max_aux_streams = 0
        print("Building static FP16 engine", flush=True)
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed")
        data = bytes(engine)
    output.write_bytes(data)
    metadata = {
        "weights_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(data).hexdigest(),
        "gpu": torch.cuda.get_device_name(), "tensorrt": trt.__version__,
        "input_shape": list(shape),
    }
    Path(str(output) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {output}", flush=True)


if __name__ == "__main__":
    main()
