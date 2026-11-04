# 推理配置与计时

## 统一开关

编辑 `supership/inference_config.py` 中 `InferenceConfig` 的默认值：

```python
backend: str = "tensorrt"
engine_path: str = "weights/supership.engine"
amp: bool = False
cuda_graph: bool = False
async_prefetch: bool = False
channels_last: bool = False
fuse_conv_bn: bool = False
profiling: bool = False
```

正式推理、分项分析和 NVTX 脚本共用配置。开启 CUDA Graph 只需把 `cuda_graph` 改为 `True`，然后直接运行命令，不再读取 `ENABLE_AMP`、`ENABLE_CUDA_GRAPH` 或 `ENABLE_PROFILING` 环境变量。优化开关默认关闭；当前后端为 TensorRT，改回 `backend="pytorch"` 即恢复原 PyTorch 基线。

`channels_last` 和 `fuse_conv_bn` 是预留项，尚未实现；设为 True 会明确报错，不会静默忽略。`profiling` 只控制分析脚本，正式推理不注册计时挂钩。

## 正式端到端计时

```bash
source activate_env.sh
bash ./benchmark_reference.sh
```

每次先完整预热 1 轮，再正式测试 3 轮；`RUNS` 仅表示正式轮数。预热结果保存为 `output/reference_submission.warmup.json`，只打印 `warmup:` 行，不计入最终 `runs_seconds` 或中位数。每轮 worker 独立启动，因此外部预热主要预热系统与文件缓存；CUDA Graph 仍在各 worker 的 READY 前用合成输入预热、捕获，不提前读取评测数据。

逐轮保留原有计时和 AP50 输出：

```text
AP50 / baseline = 0.793473 / 0.793473
```

评分在每轮推理计时结束后执行，不重复推理 baseline。`COMPARE_ACCURACY=False` 可关闭评分，`OUTPUT_FILE` 可指定结果路径。基准为 200 张参考集上的既有数值，不是官网验证集的 0.916。默认结果会覆盖，比较实验请分别保存输出。

## 模型分项分析

把配置中的 `profiling` 改为 `True` 后执行：

```bash
python scripts/profile_inference.py --runs 3 --output analysis/module-timings/submission.json
```

同样先预热 1 轮；正式轮次的分项保存在 `profile_metrics.json` 和 `profile_report.txt`。CUDA 用 Event 统计，CPU 用主机时钟；三个模块的累计时间和平均每批时间属于前向子项，不能与前向总时间重复相加。分项含冷启动和计时开销，用于定位瓶颈，正式性能看 benchmark。

## CUDA Graph 与 AMP

独立的 `CUDAGraphRunner.forward(samples)` 管理固定输入、捕获和 replay；模型结构和权重不变。末批不足、类型或设备不匹配时回退到普通前向。Graph 输出会被下一次 replay 覆盖，需先消费；CUDA Graph replay 和 TensorRT 都不执行 Python 模块挂钩，因此分析脚本只显示整体前向，不给出无效的三个模块耗时。

AMP 仅在 CUDA 前向启用 autocast，后处理输入恢复 FP32；CPU 保持普通精度。不需要 `.half()` 或 GradScaler。AMP 可能影响 AP，需同时比较精度和时间。两项优化默认均关闭，每次只改一项以便判断收益。


## 独立拷贝流预取

配置中设置 `async_prefetch: bool = True`，然后照常运行 `bash ./benchmark_reference.sh`（自动预热一轮）。单个开关同时开启 DataLoader pinned memory 和独立 CUDA H2D 流，不额外复制整批已锁页图像。关闭时仍在当前流直接 `.to(device)`。

现有 8 个 DataLoader worker 已能在 CPU 后台准备批次，但没有 GPU 输入传输重叠。新增 `DevicePrefetcher` 在当前批前向提交后上传下一批，计算流通过该批 Event 等待，并用 `record_stream` 保护跨流 GPU 张量；保留 CPU 源张量直到 DMA 完成，不修改或复用其存储。锁页源数据最多保留两批，只有生产过快时才做最旧拷贝的事件级背压；提前退出时也会安全结束未完成的拷贝。

Graph 固定输入仍只由计算流写入，预取使用独立输入张量，不会覆盖 Graph 输出；最后不足一批仍走普通前向。正式入口和两个分析脚本共用该预取器，分析时 H2D Event 记录在拷贝流。

`image_id.item()` 读取 CPU 数据，不是 GPU 同步点；结果 `.tolist()` 是 JSON 整理所需的回传同步，因此保留，并把下一批上传放在它之前。循环内没有新增全局 `torch.cuda.synchronize()`，只保留初始化及推理结束的同步。独立流提供重叠条件，实际重叠和端到端收益仍需用时间线及多轮计时判断。


## TensorRT 后端

一次性安装可选依赖并离线构建引擎（使用合成输入，不读取参考集）：

```bash
NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1 .venv/bin/python -m pip install -r requirements-tensorrt.txt
.venv/bin/python scripts/build_tensorrt.py --fp16
```

当前 `inference_config.py` 已设置 `backend="tensorrt"`，构建好引擎后直接运行 `bash ./benchmark_reference.sh`。改回 `"pytorch"` 即恢复原推理路径。引擎通过独立 `TensorRTRunner.forward(samples)` 在非默认 CUDA 流异步执行，并与调用流建立依赖，返回原有 logits/boxes，沿用原 FP32 后处理和 AP50 输出。可与异步预取配合；不会再叠加 PyTorch CUDA Graph 捕获。AMP 配置只影响 PyTorch；TensorRT 精度在构建时由 `--fp16` 决定。

默认支持 batch 1～8，最后不足一批也直接由 TensorRT 执行；更大 batch 需用 `--max-batch-size` 重新构建。引擎及其 `.engine.json` 校验信息均保存在本地，包含配置/权重指纹、GPU、TensorRT 版本和引擎校验和；配置、权重或运行版本变化后必须重建，避免误用缓存。缺少引擎时明确报错，不静默回退 PyTorch。首次构建耗时不包含在 benchmark 中，每次测试仍先完整预热一轮再正式计时。
