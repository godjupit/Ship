# 独立计时脚本

在项目根目录执行：

```bash
source activate_env.sh
python scripts/profile_inference.py
```

默认对 200 张参考图像运行一次，检测结果写入 `output/profile_submission.json`。

```bash
python scripts/profile_inference.py --runs 3 --batch-size 8
```

每轮打印 `PROFILE {...}`，时间单位为秒：

- `loader_create_wall_s`：创建加载器、读取元数据。
- `loader_wait_wall_s`：主进程等待 batch，含 worker 启动；不是全部 worker 的 CPU 工作时间之和。
- `h2d_gpu_s`：图片与原始尺寸的 CPU→GPU 传输阶段。
- `forward_gpu_s`：模型前向计算。
- `postprocess_gpu_s`：GPU 后处理。
- `results_wall_s`：结果回传、同步等待及 CPU 整理。
- `write_wall_s`：JSON 写盘及 flush/fsync。
- `h2d_bytes` / `images` / `batches`：输入传输字节数、图像数、batch 数。

GPU 阶段使用 CUDA Event，最后统一读取，不逐阶段增加同步。CPU 等待与 GPU 阶段可能重叠，不能简单相加。`total_wall_s` 只包含推理函数，端到端时间看官方计时工具打印的 `run` / `median_seconds`。

脚本复用正式入口的模型构建和加载器，独立执行计时流程；`run.sh` 和 `supership/generate_submission.py` 保持原版。
