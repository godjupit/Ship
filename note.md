## 数据传输

减少传输的数据量、提高 PCIe 传输效率、让数据传输与 GPU 计算重叠。
当前图像输入为 640 × 640 × 3 的 float32 数据，每张约 4.69 MiB。

## 分支

- `baseline-no-optimization`：基于 `7e901a6`，本轮数据传输优化前的版本。未开启 DataLoader pinned memory，使用 `.to(device)` 上传，没有复用缓冲区。
- `main`：当前优化版本，开启 DataLoader pinned memory，复用 pinned CPU / GPU 缓冲区并异步上传；仍使用 float32。
- baseline 分支只补充本说明，不引入优化代码。两个版本的差异包含 pinned memory、异步上传和缓冲区复用，不能单独归因于 pinned memory。

## 两个分支的运行命令

在项目根目录、同一个 Bash 终端依次执行。切换前先提交或妥善保存自己的代码改动。
数据、权重、`.venv` 和 `output/` 保存在本地，不随分支切换。

先定义相同的测试命令（使用正式推理入口）：

```bash
benchmark_branch() {
  .venv/bin/python supership/benchmark_submission.py \
    --image-dir data/SuperShip/reference/images \
    --meta-file data/SuperShip/reference/reference_images_meta.json \
    --output-file "output/$1/submission.json" \
    --runs "$2" \
    -- env PYTHON_BIN=.venv/bin/python BATCH_SIZE=8 DEVICE=cuda:0 bash run.sh --worker \
    > "output/$1.log" 2>&1
  cat "output/$1.log"
}
mkdir -p output
```

运行无优化版本：

```bash
git switch baseline-no-optimization
benchmark_branch baseline-warmup 1
benchmark_branch baseline 5
```

运行当前优化版本，最后停留在 main：

```bash
git switch main
benchmark_branch optimized-warmup 1
benchmark_branch optimized 5
```

只比较正式 5 轮日志末尾的 `median_seconds`，预热轮不纳入比较。
日志分别为 `output/baseline.log` 和 `output/optimized.log`，结果文件按轮次单独保存。
加速百分比 = (baseline 中位数 − optimized 中位数) / baseline 中位数 × 100%。
保持 GPU、数据、权重、batch size、worker 数一致，避免同时运行其他 GPU 任务。
建议反向再测一次以检查波动；重复前修改输出名称，避免覆盖已有记录。
历史单轮参考：正式入口 3.9902 秒，分项脚本 3.8747 秒；公平对比以本次同环境多轮结果为准。
