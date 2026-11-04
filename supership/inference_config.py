"""推理和分析脚本共用的开关；修改此文件后直接运行 benchmark 即可。"""
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    backend: str = "tensorrt"  # 改为 "pytorch" 切回原基线。
    engine_path: str = "weights/supership-576.engine"
    def validate(self):
        """校验后端及预留项，避免以为开启了实际上没有生效的优化。"""
        if self.backend not in ("pytorch", "tensorrt"):
            raise ValueError("backend must be 'pytorch' or 'tensorrt'")



# 正式推理与实验脚本共用后端设置。
INFERENCE_CONFIG = InferenceConfig()
