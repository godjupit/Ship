# 推理与实验脚本

所有命令从仓库根目录执行，先激活环境：

```bash
source activate_env.sh
```

## 正式推理

后端与引擎路径在 `supership/inference_config.py` 中设置。当前使用 TensorRT FP16，输入为 576×576，默认 batch 为 8。`run.sh` 的 `BATCH_SIZE` 控制推理批量，主 YAML 中 `val_dataloader.num_workers` 控制数据加载进程数。

```bash
bash benchmark_reference.sh
```

基准测试先预热一次，再正式运行三轮，并在计时外输出 AP50。`RUNS` 控制正式轮数，`OUTPUT_FILE` 指定输出位置，`COMPARE_ACCURACY=False` 关闭评分。比较实验应使用独立输出目录。

## 构建 TensorRT 引擎

```bash
NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1 python -m pip install -r requirements-tensorrt.txt
python scripts/build_tensorrt.py --fp16
```

构建使用 YAML 的 `eval_spatial_size` 和合成输入，不读取参考集。默认支持 batch 1～8，配置或权重变化后需重新构建。引擎加载会检查配置、权重、GPU、TensorRT 版本和文件校验和。PyTorch 后端仍用于导出及基线验证。

## 实验工具

独立试验放在 `scripts/experiments/`：

- `benchmark_batch_workers.py`：扫描 batch 和 worker 组合，复用正式推理与评分。
- `try_torch_tensorrt.py`：Torch-TensorRT 编译、合成输入误差及前向计时；依赖见同目录 `requirements-torch-tensorrt.txt`，使用独立 `.venv-torchtrt` 环境。

```bash
python scripts/experiments/benchmark_batch_workers.py --batch-sizes 1 2 4 8 --workers 0 2 4 8 --runs 3
.venv-torchtrt/bin/python scripts/experiments/try_torch_tensorrt.py --help
```

历史推理结果和日志保存在 `output/`，分析产物保存在 `analysis/`，根目录的 Nsight 捕获已归入 `analysis/nsight/`。这些生成文件不进入提交。已移除停用的 CUDA Graph、AMP、异步预取及其分析工具；旧实现可通过 Git 历史查阅。
