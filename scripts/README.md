# 推理与实验脚本

所有命令从仓库根目录执行，先激活环境：

```bash
source activate_env.sh
```

## 正式推理

默认入口 `run.sh` 使用 `supership/fast_worker.py`，配置集中在 `configs/fastest.env`，调用方的环境变量优先。当前默认 PyTorch CUDA Graph / FP16，输入 544×544，batch 8，OpenCV JPEG 缩小解码（2），top100，置信度阈值 0.05。默认 10 个解码线程；`SLURM_CPUS_PER_TASK=8` 时默认 6 个。`NUM_WORKERS` 控制解码线程数，YAML 的 DataLoader worker 数不控制 fast 流水线。

```bash
bash benchmark_reference.sh
```

基准测试先预热一次，再正式运行三轮，并在计时外输出 AP50。`RUNS` 控制正式轮数，`OUTPUT_FILE` 指定输出位置，`COMPARE_ACCURACY=False` 关闭评分。比较实验应使用独立输出目录。

`configs/dfine/dfine_hgnetv2_s_supership.yml` 的 640 尺寸用于加载原始权重；fast worker 随后根据 `INPUT_SIZE` 重建位置编码和 anchors。修改 fast 输入尺寸使用 `INPUT_SIZE`，无需 TensorRT 引擎。

```bash
INPUT_SIZE=576 IMAGE_DECODER=pil JPEG_REDUCTION=1 POST_TOP_QUERIES=300 SCORE_THRESHOLD=0 bash benchmark_reference.sh
```

上述命令使用实验中较保守的精度配置。`PIPELINE=standard` 可运行普通 PyTorch 640 FP32 路径。旧 TensorRT 推理和配套实验脚本可从 Git 历史查看；不再作为当前默认配置。

### TensorRT 544 候选

快速流水线支持可选 TensorRT 后端，仍保留现有读图、后处理、JSON 写出及 CUDA Graph。先安装 `requirements-tensorrt.txt`，用 `python scripts/build_tensorrt.py` 离线构建静态 544、batch 8 的 FP16 引擎。构建仅使用权重及合成输入。运行：

```bash
BACKEND=tensorrt RUNS=3 bash benchmark_reference.sh
```

`TRT_ENGINE` 默认 `weights/supership-544.engine`，须同时携带同名 `.json` 校验文件。运行时检查权重、引擎校验和、GPU、TensorRT 版本及输入形状；尾批补零至构建 batch，写出时仅保留真实图片。修改输入大小或 batch 后需重新构建。TensorRT 的数值结果可能与 PyTorch FP16 不同，必须另行评估 AP50。

快速流水线额外依赖见 `requirements-fast.txt`，在现有 CUDA PyTorch 环境中安装。新增流水线测试使用 pytest；完整验证运行 `python -m pytest tests -q`，原有 unittest 测试仍可运行。

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
