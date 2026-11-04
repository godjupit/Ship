"""Run timed inference and print concise accuracy comparisons."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def evaluate(split, args):
    runs = args.runs
    title = "验证集" if split == "val" else "公开参考测试集"
    image_count = 1030 if split == "val" else 200
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    benchmark_log = output / f"{split}_benchmark.log"
    runtime_log = output / f"{split}_runtime.log"
    base = output / f"{split}_submission.json"
    metadata = (
        f"data/SuperShip/{split}/annotations/instances_{split}.json"
        if split == "val"
        else "data/SuperShip/reference/reference_images_meta.json"
    )
    command = [
        sys.executable, "supership/benchmark_submission.py",
        "--image-dir", f"data/SuperShip/{split}/images",
        "--meta-file", metadata,
        "--output-file", str(base), "--runs", str(runs),
        "--init-timeout", str(args.init_timeout),
        "--run-timeout", str(args.run_timeout),
        "--", "bash", "run.sh", "--worker",
    ]
    print(f"\n=== {title}：{image_count} 张，{runs} 次 ===", flush=True)
    report = None
    with benchmark_log.open("w") as log, runtime_log.open("w") as errors:
        with subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=errors,
            text=True, bufsize=1,
        ) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("run "):
                    print(line.rstrip(), flush=True)
                elif line.startswith('{"runs_seconds"'):
                    report = json.loads(line)
            return_code = process.wait()
    if return_code != 0 or report is None:
        raise RuntimeError(f"{title}计时失败，请查看 {runtime_log}")

    ap50s = []
    for index in range(1, runs + 1):
        result = base if runs == 1 else base.with_name(f"{base.stem}.run{index}.json")
        score_log = output / f"{split}_score.run{index}.log"
        score = subprocess.run([
            sys.executable, "supership/score_submission.py",
            "--gt", f"data/SuperShip/{split}/annotations/instances_{split}.json",
            "--submission", str(result),
        ], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        score_log.write_text(score.stdout)
        match = re.search(r"^\s*AP@0\.50\s+([0-9.]+)\s*$", score.stdout, re.MULTILINE)
        if score.returncode != 0 or match is None:
            raise RuntimeError(f"第 {index} 次评分失败，请查看 {score_log}")
        ap50 = float(match.group(1))
        ap50s.append(ap50)
        print(f"第 {index} 次 AP50：{ap50:.4f}（{ap50:.2%}）", flush=True)

    seconds = report["median_seconds"]
    print(f"端到端时间中位数：{seconds:.3f} 秒；吞吐量：{image_count / seconds:.1f} 张/秒")
    baseline = 0.916 if split == "val" else 0.7929
    label = "官方验证集参考" if split == "val" else "本地参考集基线（非官方公布值）"
    print(f"{label} AP50：{baseline:.4f}")
    differences = ", ".join(f"{(value - baseline) * 100:+.2f}" for value in ap50s)
    print(f"各轮相对基线差值：{differences} 个百分点")
    print(f"检测结果：{base if runs == 1 else base.with_name(base.stem + '.run[1-' + str(runs) + '].json')}")
    print(f"计时日志：{benchmark_log}")


def main():
    parser = argparse.ArgumentParser(description="计时、评分并与基线比较，默认评估验证集和参考集。")
    parser.add_argument("split", nargs="?", choices=["all", "val", "reference"], default=os.environ.get("EVAL_SPLIT", "all"))
    parser.add_argument("--runs", type=int, default=int(os.environ.get("RUNS", "3")))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "output"))
    parser.add_argument("--init-timeout", type=float, default=float(os.environ.get("INIT_TIMEOUT", "300")))
    parser.add_argument("--run-timeout", type=float, default=float(os.environ.get("RUN_TIMEOUT", "3600")))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs 必须大于 0")
    if args.split not in ("all", "val", "reference"):
        parser.error("EVAL_SPLIT 必须为 all、val 或 reference")
    if args.init_timeout <= 0 or args.run_timeout <= 0:
        parser.error("超时秒数必须大于 0")
    os.chdir(ROOT)
    os.environ["PYTHON_BIN"] = sys.executable
    print(f"设备：{os.environ.get('DEVICE', 'cuda:0')}；batch size：{os.environ.get('BATCH_SIZE', '8')}")
    backend = os.environ.get("BACKEND", "torch")
    precision = os.environ.get("PRECISION", "fp32")
    if backend == "tensorrt" and os.environ.get("ENGINE_PATH"):
        metadata_path = Path(os.environ["ENGINE_PATH"] + ".json")
        if metadata_path.exists():
            precision = json.loads(metadata_path.read_text()).get("precision", "unknown")
    backend_label = backend + ("+cuda_graph" if os.environ.get("CUDA_GRAPH") == "1" and backend != "cuda_graph" else "")
    print(f"后端：{backend_label}；精度：{precision}")
    print("计时不含模型初始化，包含读取、预处理、推理、后处理、写盘和退出。评分在计时之外进行。")
    for split in (["val", "reference"] if args.split == "all" else [args.split]):
        evaluate(split, args)
    print("\n以上为本地评测；比赛有效性以隐藏集 AP50 达到同集官方基线的 98% 为准。")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        print(f"评测失败：{error}", file=sys.stderr)
        sys.exit(1)
