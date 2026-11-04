"""参考集 AP 评分辅助函数；不运行 baseline，也不修改计时日志格式。"""
from contextlib import redirect_stdout
import io

if __package__:
    from .score_submission import COCO, COCOeval_faster, load_detections
    from .submission import read_submission
else:
    from score_submission import COCO, COCOeval_faster, load_detections
    from submission import read_submission


# baseline.pth 在本仓库 200 张参考集上的既有 AP50，不是验证集的 0.916。
REFERENCE_AP50 = 0.7934734479978975


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
