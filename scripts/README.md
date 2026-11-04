# 推理与实验脚本

所有命令从仓库根目录执行，先激活环境：

```bash
source activate_env.sh
```

## 正式推理

默认入口 `run.sh` 使用 `supership/trt_worker.py`：独立 TensorRT/CUDA Driver 运行时，不导入 PyTorch。配置集中在 `configs/fastest.env`，调用方环境变量优先。当前预设为 FP16 引擎、512×512、batch 8、100 个解码查询、top100、置信度阈值 0.05，OpenCV RGB 缩小解码（2）。默认 10 个解码线程；`SLURM_CPUS_PER_TASK=8` 时为 6 个。

首次构建需要已有 CUDA PyTorch 环境、`requirements-fast.txt`、`requirements-tensorrt.txt` 和 CUDA Toolkit 的 `nvcc`：

```bash
python scripts/build_tensorrt.py --side 512 --num-queries 100 --output weights/supership-512-q100.engine
python scripts/build_trt_kernels.py
bash benchmark_reference.sh
```

构建仅使用原始权重和合成输入。离线部署需携带 `weights/baseline.pth`、上述 `.engine`、`weights/trt_preprocess.ptx` 及两者同名 `.json` 校验文件。运行时不调用编译器、不访问网络；检查引擎、权重、PTX、CUDA 源码校验和、GPU、TensorRT 版本和输入形状。改变输入尺寸、batch 或查询数必须重新构建引擎。`TRT_ENGINE`、`TRT_KERNELS` 可覆盖产物路径。

所有评测数据在 READY 后读取。输入和输出使用有界 pinned 缓冲区；事件分别保护 H2D 输入复用和 D2H 输出读取。CUDA Graph 包含归一化、推理和 sigmoid；CPU 线程完成 top-k、坐标换算和 JSON 序列化。尾批补零，只写出真实图像。全部任务完成、输出 fsync 并关闭后才退出。独立运行时固定使用 RGB 解码和线程化 JSON；原 PyTorch 路径的算子开关不适用于它。

基准测试先预热一次，再正式运行三轮，并在计时外输出 AP50。`RUNS` 控制正式轮数，`OUTPUT_FILE` 指定输出位置，`COMPARE_ACCURACY=False` 关闭评分。比较实验应使用独立输出目录。200 张参考集交错三轮对照：旧 544 TensorRT 路径中位数 0.21632 秒，独立 512/100-query 路径 0.16537 秒；AP50 为 0.7824626。该精度和性能不代表隐藏测试集。

保留原路径用于对照：

```bash
# 原 PyTorch 544 CUDA Graph
BACKEND=cuda_graph bash benchmark_reference.sh
# 原 TensorRT 544 引擎，由 PyTorch 管理缓冲区和后处理
BACKEND=tensorrt TRT_STANDALONE=0 INPUT_SIZE=544 TRT_ENGINE=weights/supership-544.engine bash benchmark_reference.sh
# 普通 PyTorch 640 FP32
PIPELINE=standard bash benchmark_reference.sh
```

PyTorch fast worker 的 YAML 640 尺寸用于严格加载权重，再根据 `INPUT_SIZE` 重建几何参数。`NUM_WORKERS` 控制解码线程；YAML DataLoader worker 不控制此流水线。完整验证使用 `python -m pytest tests -q`，也保留 `python -m unittest discover -s tests -v`。

历史推理结果保存在 `output/`，分析产物保存在 `analysis/`，不进入提交。544 组合改变了解码、分辨率和后处理工作量，精度与性能需一起评估；参考集结果不能保证隐藏集精度达标。

## Nsight Systems 时间线

系统自带的 Nsight Systems 2022.4 在当前 CUDA 12.4 环境解析时报 `Unknown runtime API function index: 440`，生成的报告缺少 kernel 数据。使用项目本地的 2024.6.2：

```bash
mkdir -p .local-tools/nsight-systems
curl -fL -o /tmp/supership-nsight.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2024.6.2_2024.6.2.225-1_amd64.deb
dpkg-deb -x /tmp/supership-nsight.deb .local-tools/nsight-systems
bash scripts/experiments/profile_nsys.sh
```

安装包来自 [NVIDIA 官方仓库](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/)，解压安装不会替换系统 CUDA、驱动或 PyTorch。工具本体不进入 Git。安装后重新执行 `source activate_env.sh`，`nsys` 会优先使用本地新版。

采集脚本直接运行正式推理入口，将版本、日志、`.nsys-rep`、SQLite、统计和检测结果集中保存在独立的 `analysis/nsight/时间戳/` 下，并检查 kernel 和内存传输记录非空。可通过 `NSYS_BIN` 指定其他 Nsight 路径，`BATCH_SIZE` 指定批量，`PROFILE_DIR` 指定输出目录。

报告包含初始化与合成输入预热，查看时间线时应区分启动阶段与实际图片推理。采集会增加运行开销，正式耗时仍用 `benchmark_reference.sh` 测量；有 CUDA Event 或 CPU 分段计时的旧 JSON 属于独立分析结果，不是 Nsight 报告。

新版统计项名称为 `cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum`，不要沿用旧版的 `cudaapisum,gpukernsum,gpumemtimesum`。重新统计已有报告时可执行：

```bash
source activate_env.sh
nsys stats --force-export=true \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  analysis/nsight/<采集目录>/inference.nsys-rep
```
