# SuperShip 服务器环境

在 `/home/ubuntu/SuperShip-main` 执行：

```bash
source activate_env.sh
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

- Python 3.12，项目环境 `.venv`；通过 `--system-site-packages` 复用 base 中已有的普通 Python 依赖。
- 使用 PyTorch 2.6.0+cu124 / torchvision 0.21.0+cu124，CUDA 运行时为 12.4。
- 复用服务器已下载的 CUDA 12.4 NVRTC、runtime、cuBLAS、cuFFT、cuRAND、cuSOLVER、cuSPARSE，以及校验通过的 cuSPARSELt、CUPTI、NVTX、nvJitLink，文件复制到项目环境，不依赖 `/dev/shm/juno` 持续存在。
- base 中 cuDNN、NCCL 的旧版安装记录与实际库文件不符，文件被 CUDA 13 包覆盖；项目环境重新安装兼容的 cuDNN 9.1 / NCCL 2.21.5。配套使用 Triton 3.2.0 / SymPy 1.13.1。
- 不修改系统 NVIDIA 驱动和 `/usr/local/cuda`。系统 `nvcc` 仍为 11.8；本环境配置的是 PyTorch 推理所需的 12.4 运行库，没有安装 12.4 编译工具链。
- 默认入口为 `bash run.sh --worker`，使用 `configs/fastest.env` 的 544×544 独立 TensorRT / FP16 流水线（CUDA Graph）；额外依赖见 `requirements-fast.txt`，权重为 `weights/baseline.pth`，引擎为 `weights/supership-544.engine`。普通 PyTorch 基线入口仍为 `supership/generate_submission.py`。

## 验证结果

CUDA 张量运算、baseline 权重加载、640×640 合成输入前向和后处理检查均通过。200 张公开参考图像的单次计时为 3.990192 秒，AP50 为 0.7935；该次原始计时和评分输出合并保存在 `output/baseline-check.txt`。仅安装了参考集，训练集和验证集尚未安装。

NumPy 复用 base 的 1.26.4，Matplotlib 固定为 3.9.4、ContourPy 固定为 1.3.1，避免新版 ContourPy 自动升级 NumPy 后与已有 SciPy 冲突。这些约束已写入 `requirements-cuda124.txt`。

`pip check` 仍提示继承的 base 中 Google API / gRPC 等包存在版本冲突，未调整这些与基线推理无关的应用依赖。CUDA、权重加载、前向和后处理验证均已通过。
