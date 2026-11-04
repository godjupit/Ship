"""对当前结果评分并按固定参考集基准检查 AP；不重复运行 baseline。"""
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys

from .benchmark_submission import output_path_for_run
from .score_submission import COCO, COCOeval_faster, load_detections
from .submission import read_submission


# 官网的有效性规则是“同一测试集 AP50 >= baseline 的 98%”。
# 0.916 属于验证集，下面是 baseline.pth 在本仓库 200 张参考集的实测值。
REFERENCE_BASELINE = {"mAP": 0.6794220111003091, "AP50": 0.7934734479978975,
                      "AP75": 0.7706652278753053}
RETENTION_RATIO = 0.98
RULE_SOURCE = "https://lab.cs.tsinghua.edu.cn/scc/Applications/SuperShip/#_16"


def reference_accuracy_check(metrics):
    """只判定本地参考集阈值，不宣称满足未知隐藏集的正式精度门槛。"""
    threshold = RETENTION_RATIO * REFERENCE_BASELINE["AP50"]
    return {"ap50_threshold": threshold, "passed": metrics["AP50"] >= threshold}


def score_predictions(submission, ground_truth):
    """复用官方 COCO 评分流程，返回原始精度数值而不是解析四舍五入文本。"""
    predictions = read_submission(submission, ground_truth)
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(ground_truth))
        coco_dt = load_detections(coco_gt, predictions)
        evaluator = COCOeval_faster(coco_gt, cocoDt=coco_dt, iouType="bbox",
                                   print_function=lambda *args: None)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return dict(zip(("mAP", "AP50", "AP75"), map(float, evaluator.stats[:3])))


def accuracy_difference(current, baseline):
    """差值为当前减基准；正数代表 AP 提高，数值使用 0～1 标度。"""
    return {name: current[name] - baseline[name] for name in ("mAP", "AP50", "AP75")}


def run_benchmark(output, runs, amp, log_path):
    """调用相同的正式 benchmark；关闭对比开关以防递归进入本脚本。"""
    env = dict(os.environ, COMPARE_ACCURACY="False", OUTPUT_FILE=str(output),
               RUNS=str(runs), ENABLE_AMP=amp, PYTHON_BIN=sys.executable)
    completed = subprocess.run(["bash", "./benchmark_reference.sh"], env=env,
                               capture_output=True, text=True)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    # 最后只输出一次计时汇总，避免比赛日志出现重复 JSON 汇总。
    print("\n".join(completed.stdout.strip().splitlines()[:-1]), flush=True)
    print(completed.stderr, end="", file=sys.stderr, flush=True)
    completed.check_returncode()
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main():
    """只运行当前版本，再逐轮评分，与固定 FP32 参考数值比较。"""
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    runs = int(os.environ.get("RUNS", "3"))
    if runs < 1:
        raise ValueError("RUNS must be positive")
    # 每次独立建目录，保留各轮检测、完整日志和可复查的比较报告。
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder = Path("output") / f"accuracy-compare-{timestamp}"
    folder.mkdir(parents=True)
    gt = Path("data/SuperShip/reference/annotations/instances_reference.json")
    current_output = folder / "current" / "submission.json"
    current_timing = run_benchmark(current_output, runs,
                                   os.environ.get("ENABLE_AMP", "False"), folder / "current.log")
    baseline_metrics = REFERENCE_BASELINE
    comparisons = []
    print("\n参考集精度对比（当前 − 固定 FP32 基准；不重复推理 baseline）")
    for index in range(runs):
        path = output_path_for_run(str(current_output), index, runs)
        metrics = score_predictions(path, gt)
        delta = accuracy_difference(metrics, baseline_metrics)
        check = reference_accuracy_check(metrics)
        comparisons.append({"submission": path, "metrics": metrics, "delta": delta,
                            "reference_accuracy_check": check})
        print(f"第 {index + 1} 轮：")
        for name in ("mAP", "AP50", "AP75"):
            print(f"  {name:4s} 基准 {baseline_metrics[name]:.6f}  "
                  f"当前 {metrics[name]:.6f}  差值 {delta[name]:+.6f}")
        status = "通过" if check["passed"] else "未通过"
        print(f"  本地 AP50 >= {check['ap50_threshold']:.6f}：{status}")
    print("仅为 200 张参考集检查；官网验证集基准为 0.916，隐藏集需单独评测。")
    report = {**current_timing, "baseline_metrics": baseline_metrics,
              "baseline_source": "baseline.pth on the released 200-image reference split",
              "rule_source": RULE_SOURCE, "retention_ratio": RETENTION_RATIO,
              "official_validation_baseline_ap50": 0.916,
              "accuracy_comparisons": comparisons}
    report_path = folder / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"完整比较报告：{report_path}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
