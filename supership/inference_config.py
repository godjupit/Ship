"""推理和分析脚本共用的开关；修改此文件后直接运行 benchmark 即可。"""
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    backend: str = "tensorrt"  # 改为 "pytorch" 切回原基线。
    engine_path: str = "weights/supership-576.engine"
    amp: bool = False
    cuda_graph: bool = False
    async_prefetch: bool = False  # 独立拷贝流 + DataLoader 锁页内存。
    channels_last: bool = False  # 预留，尚未实现。
    fuse_conv_bn: bool = False  # 预留，尚未实现。
    profiling: bool = False  # 仅控制分析脚本，正式推理不注册计时挂钩。

    def validate(self):
        """校验后端及预留项，避免以为开启了实际上没有生效的优化。"""
        if self.backend not in ("pytorch", "tensorrt"):
            raise ValueError("backend must be 'pytorch' or 'tensorrt'")
        for name in ("channels_last", "fuse_conv_bn"):
            if getattr(self, name):
                raise NotImplementedError(f"{name} is reserved and not implemented yet")


# 唯一配置实例；开启功能只需修改上面的默认值，不再读取同名环境变量。
INFERENCE_CONFIG = InferenceConfig()
