# 独立计时脚本

在项目根目录执行：

```bash
source activate_env.sh
ENABLE_PROFILING=True python scripts/profile_inference.py
```

默认对 200 张参考图像运行一次，检测结果写入 `analysis/profile_submission.json`。

```bash
ENABLE_PROFILING=True python scripts/profile_inference.py --runs 3 --batch-size 8
```

每轮打印中文分项表格，包含累计秒数、平均每批毫秒数和输入传输量；末尾打印每轮端到端耗时及中位数。各项对应含义：

- `loader_create_wall_s`：创建加载器、读取元数据。
- `loader_wait_wall_s`：主进程等待 batch，含 worker 启动；不是全部 worker 的 CPU 工作时间之和。
- `h2d_gpu_s`：图片与原始尺寸的 CPU→GPU 传输阶段。
- `forward_gpu_s`：模型前向计算。
- `postprocess_gpu_s`：GPU 后处理。
- `results_wall_s`：结果回传、同步等待及 CPU 整理。
- `write_wall_s`：JSON 写盘及 flush/fsync。
- `h2d_bytes` / `images` / `batches`：输入传输字节数、图像数、batch 数。

GPU 阶段使用 CUDA Event，最后统一读取，不逐阶段增加同步。CPU 等待与 GPU 阶段可能重叠，不能简单相加。`total_wall_s` 只包含推理函数，端到端时间看官方计时工具打印的 `run` / `median_seconds`。

脚本复用正式入口的模型构建和加载器，独立执行计时流程。当前只保留 worker 数调整，数据在当前 CUDA 流直接上传；分项可能重叠，不能相加。


## 模型三个模块的耗时

```bash
ENABLE_PROFILING=True python scripts/profile_inference.py --runs 3 --output analysis/module-timings/submission.json
```

在“模型前向计算”下查看 Backbone（特征提取）、Encoder（特征融合）、Decoder（检测预测）的累计秒数和平均每批毫秒数。脚本临时注册模块执行前后的挂钩，不修改模型 forward；CUDA Event 记录当前 GPU 流，最后统一读取，CPU 使用主机时钟。对应 JSON 字段为 `backbone_gpu_s`、`encoder_gpu_s`、`decoder_gpu_s`（CPU 后缀为 `_wall_s`）。

三个模块是模型前向的子项，不能与前向总时间重复相加。分项包含首批冷启动、CPU 提交间隔及计时挂钩开销，用于定位瓶颈；正式端到端性能仍使用 `bash ./benchmark_reference.sh`。相同输出目录会覆盖旧报告。


## 计时开关

`ENABLE_PROFILING` 默认是 `False`；设为 `True`（也支持 `1`、`yes`）才会注册模块挂钩、创建 CUDA Event 和统计分项。关闭后只保留 benchmark 的端到端时间，JSON 标记 `profiling_enabled: false`，不输出虚假的零耗时。正式 `bash ./benchmark_reference.sh` 始终不使用分析脚本，因此不受该开关影响。


## FP16 自动混合精度

默认 `ENABLE_AMP=False`；`True`、`1` 或 `yes` 在 CUDA 上开启。CPU 自动保持原精度。正式入口和两个分析脚本共用 `forward_model()`：只在前向使用 autocast，类别输出和框坐标转回 FP32 后处理；不需要 `.half()` 或 GradScaler。

```bash
ENABLE_AMP=False PYTHON_BIN=.venv/bin/python RUNS=3 bash ./benchmark_reference.sh
ENABLE_AMP=True PYTHON_BIN=.venv/bin/python RUNS=3 bash ./benchmark_reference.sh
# 分析 AMP 下三个模块的耗时：
ENABLE_AMP=True ENABLE_PROFILING=True python scripts/profile_inference.py --runs 3 --output analysis/amp-modules/submission.json
```

两组先分别预热，并单独保存输出，避免默认文件被覆盖。AMP 会产生数值差异，需用 `bash ./score_reference.sh 对应结果.json` 比较 AP，不能只看速度。`ENABLE_PROFILING` 与 `ENABLE_AMP` 独立，正式 benchmark 不加载分项计时代码。


## 推理后检查参考集精度

```bash
PYTHON_BIN=.venv/bin/python RUNS=3 bash ./benchmark_reference.sh
```

只运行当前版本，不额外运行 baseline。每轮推理结束后用参考集真实标注评分，打印 mAP、AP50、AP75 与固定参考基准的差值，以及本地 AP50 门槛是否通过。评分不计入推理时间，记录保存到 `output/accuracy-compare-时间戳/comparison.json`，不会覆盖默认输出。

官网规定同一隐藏测试集上 AP50 不低于 baseline 的 98%；0.916 是验证集基准，不能用于判断 200 张参考集。本地使用已有 FP32 基准 AP50=0.7934734479978975，参考集阈值约 0.777604。通过本地检查不代表通过隐藏集评测；数据集和权重更换后需重新确定对应基准。

`bash ./benchmark_reference.sh` 默认逐轮打印“当前 AP50 / baseline”及百分比；无需额外开启对比开关。设 `COMPARE_ACCURACY=False` 可只计时、不评分，此模式可通过 `OUTPUT_FILE` 指定输出位置。默认比较模式保留每轮计时，并且只打印一次最终计时 JSON 汇总。
