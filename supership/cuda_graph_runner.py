"""固定形状 CUDA Graph 前向；不读取数据集，不修改模型结构。"""
import torch


class CUDAGraphRunner:
    """捕获固定形状的前向；输入和输出显存保持不变，供后续批次重放。"""

    def __init__(self, eager_forward, device, batch_size):
        self.eager_forward = eager_forward
        self.device = torch.device(device)
        with torch.inference_mode():
            # 仅使用合成输入，初始化期间不读取评测图片或元数据。
            self.input = torch.zeros((batch_size, 3, 640, 640), device=self.device)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.eager_forward(self.input)
            stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            # 捕获同一个普通前向 callable，不改变原模型结构或权重。
            with torch.cuda.graph(self.graph, stream=stream):
                self.outputs = self.eager_forward(self.input)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)

    def supports(self, samples):
        """不足一批或数据类型/设备不符时回退，避免错误重放固定形状图。"""
        return (samples.shape == self.input.shape and samples.dtype == self.input.dtype
                and samples.device == self.input.device)

    def forward(self, samples):
        """满批重放，不匹配时普通前向；返回值需在下一次调用前消费。"""
        if not self.supports(samples):
            return self.eager_forward(samples)
        self.input.copy_(samples)
        self.graph.replay()
        return self.outputs
