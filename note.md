## 数据传输

减少传输的数据量、提高 PCIe 传输效率、让数据传输与 GPU 计算重叠。
当前图像输入为 640 × 640 × 3 的 float32 数据，每张约 4.69 MiB。

## Pinned memory
pinned memory 是锁页的 CPU 内存，可用于异步 DMA 上传到 GPU 显存；不是让 GPU 直接读取 CPU cache。


可以。当前 pinned memory 只提升约 0.84%，建议优先减少传输量和 CPU 整理开销。

| 优先级 | 优化方向 | 修改位置 | 主要作用 |
|---|---|---|---|
| 1 | 按整个 batch 回传检测结果 | generate_submission.py 中 .tolist() 附近 | 当前每张图分别回传 boxes、scores、labels，尝试合并回传，减少小额拷贝和同步 |
| 2 | 上传 uint8，在 GPU 转 float32 并除以 255 | create_loader() 的图片转换和模型调用前 | 图片上传量降至原来的 1/4，同时减轻 CPU 浮点转换工作 |
| 3 | 调整 DataLoader worker 数 | configs/dataset/supership_detection.yml | 分别测试 2、4、8 个 worker，找出当前 CPU/GPU 配置的合适值 |
| 4 | 调整 batch size | 运行参数 | 测试 8、16、32，可能提高 GPU 吞吐，但会增加显存占用 |
| 5 | 预取下一批，使用独立 CUDA 拷贝流 | 推理循环 | 尝试让下一批上传与当前批计算重叠，需要处理流同步和数据生命周期 |
| 6 | 混合精度推理 | 模型前向附近 | 可能降低计算与显存开销，需要重新验证精度 |

我建议先做第 1 项：批量回传结果。 它不用改变输入精度和预处理逻辑，比较容易验证结果是否完全一致。然后做 uint8 上传，直接减少主要的数据传输量。

## 独立 CUDA 拷贝流

正式入口和两个分析脚本共用 `DevicePrefetcher`。先上传首批，当前批计算提交后预取下一批；拷贝流通过 Event 通知计算流，输入张量使用 `record_stream` 管理跨流生命周期。CPU 推理保留同步设备转换路径，不使用手动复用缓冲区。

测试使用正式入口：

```bash
PYTHON_BIN=.venv/bin/python RUNS=1 bash ./benchmark_reference.sh
PYTHON_BIN=.venv/bin/python RUNS=3 bash ./benchmark_reference.sh
```

首条用于预热，第二条比较端到端中位数。脚本默认输出会覆盖，需另存每次测试记录。
分项分析中的 H2D Event 记录在独立拷贝流；loader_wait 统计实际取数等待，包含启动及预取阶段。各阶段可能重叠，不能直接相加。

本轮正式入口测试（两组各预热 1 轮、正式 3 轮，batch_size=8，val num_workers=8）：
- 直接异步上传中位数：3.9395 秒。
- 独立拷贝流中位数：3.9786 秒，约慢 0.99%；其中一轮为 5.6056 秒，存在明显波动，尚未证明有加速。
- 两组 6 份检测输出完全一致，每份 60000 条。
- 报告：`output/cuda-stream-20261104-165010/comparison.json`。
- 本轮未采集 Nsight 时间线，因此尚未确认实际传输与计算重叠程度。

## 修改约定

函数和关键同步位置添加简短说明注释；每次功能修改验证后创建 Git 提交。提交说明记录测试命令、参数、各轮耗时、中位数、输出一致性及测试限制，不把未验证的改动描述为加速。
