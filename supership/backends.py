"""Fixed-batch CUDA Graph and TensorRT runtimes with padded tail batches."""

from pathlib import Path
import hashlib
import json

import torch


class CUDAGraphRuntime:
    def __init__(self, model, batch_size, device, precision="fp32"):
        from supership.generate_submission import model_forward

        if not str(device).startswith("cuda"):
            raise ValueError("CUDA Graph requires a CUDA device")
        # Captured kernels reference model/engine weights and context buffers.
        self.model = model
        self.batch_size = batch_size
        self.inputs = torch.zeros(batch_size, 3, 640, 640, device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                model_forward(model, self.inputs, precision)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph, stream=stream):
            result = model_forward(model, self.inputs, precision)
            self.outputs = {name: result[name] for name in ("pred_logits", "pred_boxes")}
        torch.cuda.synchronize(device)

    def __call__(self, images):
        count = images.shape[0]
        if not 0 < count <= self.batch_size:
            raise ValueError("input batch exceeds CUDA Graph batch size")
        if count < self.batch_size:
            self.inputs.zero_()
        self.inputs[:count].copy_(images)
        self.graph.replay()
        return {name: value[:count] for name, value in self.outputs.items()}


class TensorRTRuntime:
    def __init__(self, engine_path, batch_size, device, weights_path):
        import tensorrt as trt

        if not str(device).startswith("cuda"):
            raise ValueError("TensorRT requires a CUDA device")
        metadata = json.loads(Path(str(engine_path) + ".json").read_text())
        if metadata.get("weights_sha256") != hashlib.sha256(Path(weights_path).read_bytes()).hexdigest():
            raise ValueError("TensorRT engine was built from different weights; set MODEL_WEIGHTS correctly")
        if metadata.get("engine_sha256") != hashlib.sha256(Path(engine_path).read_bytes()).hexdigest():
            raise ValueError("TensorRT engine checksum mismatch")
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.batch_size = batch_size
        self.buffers = {}
        dtype_map = {trt.float32: torch.float32, trt.float16: torch.float16,
                     trt.int32: torch.int32, trt.int64: torch.int64}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape) or shape[0] != batch_size:
                raise ValueError(f"Engine tensor {name} shape {shape} does not match batch={batch_size}")
            self.buffers[name] = torch.empty(shape, device=device,
                                           dtype=dtype_map[self.engine.get_tensor_dtype(name)])
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
        if set(self.buffers) != {"images", "pred_logits", "pred_boxes"}:
            raise ValueError("Engine must have images, pred_logits, pred_boxes tensors")
        self.execution_stream = torch.cuda.Stream(device=device)

    def __call__(self, images):
        count = images.shape[0]
        if not 0 < count <= self.batch_size:
            raise ValueError("input batch exceeds TensorRT batch size")
        caller = torch.cuda.current_stream(images.device)
        # TensorRT synchronizes enqueueV3 on the default stream. Use a private
        # stream for eager calls, and the capture stream during CUDA Graph capture.
        stream = caller if caller.cuda_stream else self.execution_stream
        separate_stream = stream.cuda_stream != caller.cuda_stream
        if separate_stream:
            stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            if count < self.batch_size:
                self.buffers["images"].zero_()
            self.buffers["images"][:count].copy_(images)
            if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            outputs = {name: self.buffers[name][:count].float()
                       for name in ("pred_logits", "pred_boxes")}
            images.record_stream(stream)
        if separate_stream:
            caller.wait_stream(stream)
        return outputs
