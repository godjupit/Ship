# 推理与实验脚本

`main` 汇总已验证实现；`optimize/throughput-4300` 与 `optimize/reference-200` 分别保存两个目标的版本。最新为CPU YUV + GPU融合颜色/缩放，加9bit Huffman查表与PGO解码库。五轮正式入口中位数2.363127秒，同轮旧库2.574512秒（快8.21%，五轮均更快）；200张为0.154298秒，同轮旧200分支0.164358秒。输出与前一YUV方案一致。实验记录在主工作目录 `实验.md`。

所有命令从仓库根目录执行，先激活环境：

```bash
source activate_env.sh
```

## 正式推理

默认入口 `run.sh` 使用 `supership/trt_worker.py`：独立 TensorRT/CUDA Driver 运行时，不导入 PyTorch。配置集中在 `configs/fastest.env`，调用方环境变量优先。当前预设为 FP16 引擎、544×544、batch 8、原模型 300 个解码查询、top100、置信度阈值 0.05，CPU YUV缩小解码（2），GPU融合颜色转换、缩放、布局转换及归一化。默认 `OPENBLAS_NUM_THREADS=1`，避免未参与后处理的 BLAS 工作线程；调用方可覆盖。当前 10 CPU 环境默认 9 个解码线程；`SLURM_CPUS_PER_TASK=8` 时为 6 个。

首次构建需要已有 CUDA PyTorch 环境、`requirements-fast.txt`、`requirements-tensorrt.txt` 和 CUDA Toolkit 的 `nvcc`：

```bash
python scripts/build_tensorrt.py --side 544 --output weights/supership-544.engine
python scripts/build_trt_kernels.py
python scripts/build_trt_kernels.py --source trt_yuv
# Source checkout must be unmodified at this exact revision; NASM is required.
git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git .local-tools/libjpeg-source
git -C .local-tools/libjpeg-source checkout 6c87537f60941f3c265c339fe60d1e31d2a42ccf
python scripts/build_jpeg_decoder.py --source .local-tools/libjpeg-source
bash benchmark_reference.sh
```

构建仅使用原始权重和合成输入。离线部署需携带 `weights/baseline.pth`、`weights/supership-544.engine`、`weights/trt_yuv.ptx` 及引擎/PTX同名 `.json` 校验文件。运行时不调用编译器、不访问网络；检查引擎、权重、PTX、CUDA 源码校验和、GPU、TensorRT 版本和输入形状。改变输入尺寸、batch 或查询数必须重新构建引擎。`TRT_ENGINE`、`TRT_KERNELS` 可覆盖产物路径。

JPEG运行库另需携带 `weights/libturbojpeg-pgo.so`、同名`.json`、`.LICENSE.md`和`.README.ijg`。PGO构建只读取2405张公开训练图，不使用评测图；需要GCC/G++、CMake和NASM。构建目录必须是新目录（`--work-dir`），详情在目录内`build.log`；不要与性能测试同时编译。默认针对Zen2/EPYC7402，其他CPU需确认指令兼容并重建复测。推理不需要训练集、源码、编译器或PGO数据。`TURBOJPEG_LIBRARY=bundled`切回原wheel解码器；设置其他路径时必须提供同名SHA256清单。

所有评测数据在 READY 后读取。输入和输出使用有界 pinned 缓冲区；事件分别保护 H2D 输入复用和 D2H 输出读取。CUDA Graph 包含缩放、归一化、推理和 sigmoid；CPU 线程完成 top-k、坐标换算和 JSON 序列化。尾批补零，只写出真实图像。全部任务完成、输出 fsync 并关闭后才退出。独立运行时使用线程化 JSON，`STANDALONE_DECODER=yuv` 为默认；`opencv`/`rgb` 可用于RGB对照；原 PyTorch 路径的算子开关不适用于它。


GPU 缩放使用有界 packed RGB/YUV 输入：每图的宽、高、字节偏移和格式在槽头，解码直接写入 pinned 子区域。`MAX_DECODE_PIXELS` 默认等于网络输入面积（544²），较大的解码图像回退到原 CPU resize；不会按整套图片分配内存。`GPU_RESIZE=0` 强制使用该 CPU 路径用于兼容对照。尾批描述符清零，GPU 为填充图像写零。

YUV路径支持4:2:0，其他格式回退RGB/OpenCV；每图仍预留RGB容量，当前不减少整个槽的上传跨度。YUV色度重建改变像素，验证集AP50=0.905448405，参考集AP50=0.778955797，均超过各自基线98%。新依赖 `turbojpeg==0.0.2` 固定于 `requirements-tensorrt.txt`，DALI不是运行依赖。

GPU 插值使用定点系数并在归一化前量化到 uint8；与 OpenCV 的 SIMD 舍入可能相差 1 个灰度级，因此这是经过 AP 验证的数值改动，不宣称输出与旧实现逐字节一致。单测覆盖奇数尺寸、边界、恒等尺寸、单像素和零尾批。模型结构、权重、300 个查询及 TensorRT 引擎不变。

默认在 READY 前使用合成零输入执行一次已实例化的 CUDA Graph，使首次 Graph 上传和执行发生在初始化阶段。`TRT_GRAPH_WARMUP=0` 可关闭此步骤用于对照；不会提前访问评测元数据或图像。

基准测试先预热一次，再正式运行三轮，并在计时外输出 AP50。`RUNS` 控制正式轮数，`OUTPUT_FILE` 指定输出位置，`COMPARE_ACCURACY=False` 关闭评分。比较实验应使用独立输出目录。历史 512/100-query 引擎虽然在参考集上较快，但完整验证集 AP50 仅 0.892285，低于基线 0.917040 的 98% 门槛，因此已撤回默认配置。本分支融合缩放后的完整验证集 AP50 为 0.905387，200 张参考集为 0.781422。上述精度不代表隐藏测试集。

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

## 固定优化流程

1. 保存当前 Git 提交、配置、引擎校验和和独立输出目录，检查没有其他 GPU 任务。先用正式入口记录现有方案。
2. 性能优先用 4,300 张 `throughput4300`：先预热一轮，再正反顺序交错运行基线和候选各至少五轮，保留全部耗时、中位数、输出及校验值。200 张只归档已测最快配置、时间和精度，不做专门调参。不能用 profiler 内的耗时宣称加速。
3. 用分阶段 CPU 计时与完整 GPU 时间线确定瓶颈，再改变一个主要因素。报告必须含有 CUDA kernel 和 memcpy 记录；空报告视为采集失败。
4. 数学路径未变的改动检查输出逐字节一致；改了解码、数值或引擎的方案，分别在原始 1,030 张验证集和 200 张参考集重跑并评分。各自与原始 640 FP32 基线比较，AP50 至少保留 98%。不能用重复的 4,300 张图统计精度。
5. 验证空输入、图片顺序、缓冲复用、尾批、READY 前不访问评测数据、GPU/线程完成、输出 fsync/关闭及正常退出。运行适用的测试和 `git diff --check`。
6. 只有可复现收益且精度合格才调整默认配置；失败尝试保留独立日志，不留在默认推理路径。文档记录条件、每轮结果、误差、局限和复现命令，提交已验证代码；不改用户维护的 `note.md`。

## 2026-11-05 流水线实验

RTX 4090、当前 10 CPU 环境，544 FP16、batch 8、10 个解码线程、深度 3；每组先预热一次，再交错测五轮完整验证集（1,030 张），使用正式 worker 计时到退出。除表中变量外配置相同，没有更改模型结构或权重。

| 方案 | 端到端中位数（秒） |
| --- | ---: |
| 原实现 | 0.789518 |
| 初始化阶段执行 Graph | 0.778298 |
| 独立 H2D 流、分槽设备输入、Graph 预热 | 0.792266 |
| 16 个解码线程 | 0.939115 |
| 缓冲深度 6 | 0.795985 |
| 重新构建 batch 16 引擎 | 0.798877 |

仅保留 Graph 初始化调整。额外七轮交错复测：参考集 0.190102 → 0.189329 秒；验证集 0.822047 → 0.803739 秒。差距小且有波动，不据此宣称显著加速。两套数据各轮输出均逐字节一致，验证集 AP50 保持 0.905842。batch 16 的 AP50 为 0.905667，但没有速度收益，未替换默认引擎。

另用 CUDA 11.8 nvJPEG GPU hybrid、batch 128 测试独立解码：三轮中位数 0.981124 秒；OpenCV 缩小解码加缩放为 0.656063 秒。两者都计入读文件；nvJPEG 尚未计入缩放、推理，且解码数值路径不同。该探测不是完整 nvJPEG 推理实现的端到端比较，未接入默认 worker。

原始记录与可复跑脚本保存在本地 `output/pipeline-rework/`：`matrix.py`、`matrix-summary.json`、`confirm-*-summary.json`、`decode_probe.py`、`accuracy.json` 和 `tests.log`。独立传输流试验补丁为 `experimental-overlap.patch`，未保留在生产代码中。batch 16 引擎可用 `python scripts/build_tensorrt.py --side 544 --batch-size 16 --output weights/supership-544-b16.engine` 重建。测试命令 `python -m unittest discover -s tests -v`：74 项，跳过 1 项，其余通过；包含空输入、READY 协议、缓冲复用与尾批数值一致性。另执行 `python -m pytest tests -q`：187 项通过，1 项跳过。

## 4,300 张规模的性能回归

精度使用原始 1,030 张验证集与 200 张参考集；复制出的 4,300 张负载仅用于测速。使用固定种子 20261105，从上述 1,230 张图像均衡重复采样并打乱，生成 4,300 个独立文件和唯一 ID，占约 1.2 GiB；包含 3,595 个验证图像副本和 705 个参考图像副本。`provenance.json` 保存来源，元数据不包含标注。重复内容及预热后的文件缓存不能代表隐藏集分布或冷盘性能。

```bash
source activate_env.sh
# 首次生成；已有目录会拒绝覆盖。
python scripts/experiments/make_throughput_dataset.py
# 此入口不对复制数据计算 AP。
PYTHON_BIN=.venv/bin/python RUNS=5 OUTPUT_FILE=output/throughput4300_submission.json \
  bash scripts/experiments/benchmark_throughput.sh
```

`THROUGHPUT_DATA` 可指定其他相同布局的数据目录；后端和流水线变量仍由 `run.sh` 接收。生成和测速均在仓库根目录执行。基准覆盖全部读图、预处理、推理、后处理、fsync 和进程退出，每组预热一次后测五轮，正反顺序交错执行。当前环境实测秒数如下：

| 方案 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 第 4 轮 | 第 5 轮 | 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原实现 | 3.1338 | 3.0680 | 3.2165 | 3.1079 | 3.0673 | 3.1079 |
| Graph 预热（保留） | 3.0537 | 3.0427 | 3.0483 | 3.1243 | 3.0687 | 3.0537 |
| 独立传输流 + Graph 预热 | 3.2677 | 3.0949 | 3.0425 | 3.0475 | 3.0719 | 3.0719 |
| 6 个解码线程 | 4.3369 | 4.0291 | 3.9643 | 3.9154 | 4.0344 | 4.0291 |
| 16 个解码线程 | 3.7327 | 3.6658 | 3.8458 | 3.7973 | 3.7799 | 3.7799 |
| 缓冲深度 6 | 3.0560 | 3.0736 | 3.1004 | 3.1231 | 3.1525 | 3.1004 |
| batch 16 | 3.0449 | 3.9585 | 3.0752 | 3.1244 | 3.1182 | 3.1182 |

Graph 预热方案中位数比原实现低约 1.7%，收益有限。除 batch 16 外，所有方案五轮输出逐字节一致（140,854 条检测）；batch 16 的输出在其各轮之间一致（140,729 条检测）。独立新测速入口的额外冒烟检查也通过，输出与对照一致。原始逐轮数据在 `output/pipeline-rework/matrix4300-summary.json`，执行快照为 `matrix4300.py`；这些本地生成产物不提交。

精度重新运行原始 640 FP32 基线，并在两套未复制的原始数据上评分：

| 数据集 | 640 基线 AP50 | 当前 batch 8 AP50 | 基线保留比例 | batch 16 AP50 |
| --- | ---: | ---: | ---: | ---: |
| 验证集 1,030 张 | 0.917040 | 0.905842 | 98.7789% | 0.905667 |
| 参考集 200 张 | 0.793473 | 0.781692 | 98.5152% | 0.780920 |

两套公开数据上均超过基线 98% 的门槛；这不是隐藏集保证。数值与评分执行快照保存在 `accuracy-both-splits.json` 和 `check_accuracy.py`。

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

采集脚本直接运行正式推理入口，将版本、日志、`.nsys-rep`、SQLite、统计和检测结果集中保存在独立的 `analysis/nsight/时间戳/` 下，并检查 kernel 和内存传输记录非空。可通过 `NSYS_BIN` 指定其他 Nsight 路径，`BATCH_SIZE` 指定批量，`PROFILE_DIR` 指定输出目录，`PROFILE_IMAGES` / `PROFILE_META` 指定数据集。

独立 worker 使用 `os._exit()`，因此 profiling 入口需要在退出前显式结束采集。`profile_nsys.sh` 设置 `PROFILE_NSYS=1`，经 `run.sh` 启动 `profile_worker.py`：初始化和合成预热完成后，在请求处理前调用 `cuProfilerStart()`，请求结束后通过 `finally` 调用 `cuProfilerStop()` 刷新采集。Nsight 使用 `--capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node`。正式推理不设置此变量，仍直接启动原 worker。

修复后的报告只含请求阶段，不含模型初始化。采集会增加运行开销，正式耗时用 `benchmark_throughput.sh` 测量。此脚本当前要求独立 TensorRT worker；其他后端需要自己的采集范围。4,300 张验证报告 `analysis/nsight/throughput4300-fixed/inference.nsys-rep` 包含 168,932 条 kernel、1,614 条 memcpy，符合 538 个批次；旧的 `throughput4300` 报告缺少 CUDA 数据，不能用于结论。

```bash
PROFILE_DIR=analysis/nsight/another-4300-run \
PROFILE_IMAGES=data/SuperShip/throughput4300/images \
PROFILE_META=data/SuperShip/throughput4300/images_meta.json \
  bash scripts/experiments/profile_nsys.sh
```

另有 `scripts/experiments/profile_standalone.py` 对生产函数加计时包装，分别记录读取/解码、缩放、序列化的线程 CPU 时间，以及 CUDA API 的主机等待时间。`PROFILE_CUDA_EVENTS=1` 增加 GPU Event 时间线，`PROFILE_JSON` 指定 JSON 路径。多线程累计 wall time 彼此重叠，不能相加当作端到端时间。

```bash
source configs/fastest.env
PROFILE_CUDA_EVENTS=1 PROFILE_JSON=analysis/standalone-4300-events.json \
  python scripts/experiments/profile_standalone.py \
  < analysis/nsight/throughput4300-fixed/request.json
```

4,300 张诊断中：读取/解码约 22.5 个 CPU 秒、缩放约 3.9 个 CPU 秒、序列化约 0.88 个 CPU 秒；CUDA Event 测得 Graph 约 1.476 秒、H2D 约 0.243 秒。Nsight 交叉验证 kernel 总时长约 1.524 秒。采集本身会影响时间线空闲比例，不能将 profiler 的占用率当作无采集时的 GPU 利用率。当前应优先优化 CPU 图像处理，而非传输流。

新版统计项名称为 `cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum`，不要沿用旧版的 `cudaapisum,gpukernsum,gpumemtimesum`。重新统计已有报告时可执行：

```bash
source activate_env.sh
nsys stats --force-export=true \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  analysis/nsight/<采集目录>/inference.nsys-rep
```
