"""TensorRT 10 Runner：复用 PyTorch CUDA 张量，在非默认流异步执行。"""
import hashlib
import json
from pathlib import Path
import sys

import torch


def model_fingerprint(yaml_config, weights_path):
    """只绑定网络与权重；修改 worker、数据路径或计时参数不必重建引擎。"""
    name = yaml_config.get("model")
    keys = {"model", "num_classes", "eval_spatial_size"}
    if name:
        keys.add(name)
        for component in ("backbone", "encoder", "decoder"):
            target = yaml_config.get(name, {}).get(component)
            if isinstance(target, str):
                keys.add(target)
    network_config = {key: yaml_config[key] for key in keys if key in yaml_config}
    digest = hashlib.sha256(json.dumps(network_config, sort_keys=True).encode())
    with open(weights_path, "rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensorrt():
    """仅启用 TensorRT 时导入；PyTorch 基线不依赖 TensorRT。"""
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("Install requirements-tensorrt.txt before using TensorRT") from error
    return trt


def make_logger(trt):
    """TensorRT 日志写 stderr，保持比赛标准输出格式。"""
    class Logger(trt.ILogger):
        def __init__(self):
            super().__init__()

        def log(self, severity, message):
            if severity <= trt.ILogger.Severity.WARNING:
                print(f"TensorRT: {message}", file=sys.stderr, flush=True)

    return Logger()


class TensorRTRunner:
    """动态 batch engine；forward 返回原后处理需要的两个 FP32 输出。"""

    def __init__(self, engine_path, device, fingerprint=None):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TensorRT requires CUDA; use backend='pytorch' for CPU")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        path = Path(engine_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run .venv/bin/python scripts/build_tensorrt.py --fp16")
        self.trt = load_tensorrt()
        self.logger = make_logger(self.trt)
        metadata = json.loads(Path(str(path) + ".json").read_text())
        if fingerprint is not None and metadata["model_fingerprint"] != fingerprint:
            raise ValueError("TensorRT engine does not match current config/weights; rebuild it")
        if metadata["tensorrt_version"] != self.trt.__version__:
            raise ValueError("TensorRT engine version changed; rebuild it")
        engine_data = path.read_bytes()
        if hashlib.sha256(engine_data).hexdigest() != metadata["engine_sha256"]:
            raise ValueError("TensorRT engine checksum changed; rebuild it")
        with torch.cuda.device(self.device):
            if metadata["gpu_name"] != torch.cuda.get_device_name(self.device):
                raise ValueError("TensorRT engine was built for another GPU; rebuild it")
            self.trt.init_libnvinfer_plugins(self.logger, "")
            self.runtime = self.trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("TensorRT failed to deserialize engine")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT failed to create execution context")
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if names != {"images", "pred_logits", "pred_boxes"}:
            raise ValueError(f"Unexpected engine inputs/outputs: {sorted(names)}")
        for name in names:
            if self.engine.get_tensor_dtype(name) != self.trt.float32:
                raise ValueError(f"Engine I/O must be FP32: {name}")
        self.precision = metadata["precision"]
        self.max_batch = metadata["max_batch_size"]
        self.outputs = {}
        self.shape = None
        self.stream = torch.cuda.Stream(device=self.device)
        with torch.inference_mode():
            # 初始化阶段只使用合成输入，提前完成 context/输出缓冲区初始化。
            self.forward(torch.zeros((self.max_batch, 3, 640, 640), device=self.device))

    def forward(self, samples):
        """同一计算流顺序消费输出；下次调用可能覆盖当前输出，不能跨批保存引用。"""
        if samples.device != self.device or samples.dtype != torch.float32:
            raise ValueError("TensorRT expects FP32 input on its configured CUDA device")
        shape = tuple(samples.shape)
        if len(shape) != 4 or shape[1:] != (3, 640, 640) or not 1 <= shape[0] <= self.max_batch:
            raise ValueError(f"Engine expects batch 1..{self.max_batch} of 3x640x640 images")
        with torch.cuda.device(self.device):
            samples = samples.contiguous()
            consumer = torch.cuda.current_stream(self.device)
            # TensorRT 在默认流 enqueue 会隐式同步；改用非默认流，显式承接输入依赖。
            self.stream.wait_stream(consumer)
            with torch.cuda.stream(self.stream):
                if shape != self.shape:
                    if not self.context.set_input_shape("images", shape):
                        raise RuntimeError(f"TensorRT rejected input shape {shape}")
                    self.outputs = {
                        name: torch.empty(tuple(self.context.get_tensor_shape(name)),
                                          dtype=torch.float32, device=self.device)
                        for name in ("pred_logits", "pred_boxes")
                    }
                    self.shape = shape
                if not self.context.set_tensor_address("images", samples.data_ptr()):
                    raise RuntimeError("TensorRT rejected input buffer")
                for name, tensor in self.outputs.items():
                    if not self.context.set_tensor_address(name, tensor.data_ptr()):
                        raise RuntimeError(f"TensorRT rejected output buffer {name}")
                if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
                    raise RuntimeError("TensorRT asynchronous inference failed")
                samples.record_stream(self.stream)
            consumer.wait_stream(self.stream)  # 后处理只等待这次引擎执行，不做 CPU 全局同步。
            for tensor in self.outputs.values():
                tensor.record_stream(consumer)
        return self.outputs
