## 当前生效版本

只保留 `configs/dataset/supership_detection.yml` 的 worker 调整：train 和 val 均为 8，推理使用 val。已撤销 pinned memory、异步上传、独立 CUDA 拷贝流、手动缓冲区和集中回传。图片仍为 float32，推理在当前流直接 `.to(device)`。

运行：`PYTHON_BIN=.venv/bin/python RUNS=3 bash ./benchmark_reference.sh`。

以下为历史实验记录，其中优化描述不代表当前实现；原测试数据保留在 Git 提交说明及本地 output 中。

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


bash ./benchmark_reference.sh

## 现在模型前向最费时间
使用 fp16 混合{"runs_seconds":[3.957039952,3.777571782,3.806810396],"median_seconds":3.806810396}

算是有点效果

模型的原始输入
640 * 640 * 3
之后网络会通过卷积提取特征原始输入：640 × 640

3 个通道（RGB）

卷积 + 下采样
320 × 320 → 160 × 160

S3：80 × 80

256 个特征通道

S4：40 × 40

512 个特征通道

S5：20 × 20

1024 个特征通道
最后的图片
S3 = [256, 80, 80]
S4 = [512, 40, 40]
S5 = [1024, 20, 20]

使用json 一次写入减少0.5s
主要不知道这个代码的底层一点点
