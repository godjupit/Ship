"""Score a COCO-format submission against released COCO ground truth.

Usage:
  python supership/score_submission.py \
    --gt data/SuperShip/reference/annotations/instances_reference.json \
    --submission output/reference_submission.json
"""
import argparse
import copy

from faster_coco_eval import COCO, COCOeval_faster

if __package__:
    from .submission import read_submission
else:
    from submission import read_submission


def load_detections(coco_gt, predictions):
    annotations = []
    for index, detection in enumerate(predictions, 1):
        bbox = list(detection["bbox"])
        if bbox[2] <= 0 or bbox[3] <= 0:
            # Keep the detection and its ranking. A zero-area evaluation-only
            # box has zero IoU and is not ignored in COCO's area=all metric.
            bbox = [0.0, 0.0, 0.0, 0.0]
        annotations.append({
            "id": index, "image_id": detection["image_id"],
            "category_id": detection["category_id"], "bbox": bbox,
            "score": detection["score"], "area": bbox[2] * bbox[3], "iscrowd": 0,
        })
    # Build a clean result dataset so optional user fields cannot change
    # evaluation (for example, ignore, drop, area or segmentation).
    result = COCO()
    result.dataset = {
        "images": copy.deepcopy(coco_gt.dataset["images"]),
        "categories": copy.deepcopy(coco_gt.dataset["categories"]),
        "annotations": annotations,
    }
    result.createIndex()
    return result


def per_class_ap(coco_eval):
    """AP@[.5:.95] per category (area=all, maxDets=100)."""
    precision = coco_eval.eval['precision']  # [T, R, K, A, M]
    names = {c['id']: c['name'] for c in coco_eval.cocoGt.loadCats(coco_eval.params.catIds)}
    out = {}
    for k, cat_id in enumerate(coco_eval.params.catIds):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        out[f'{cat_id}:{names[cat_id]}'] = float(p.mean()) if p.size else float('nan')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='COCO ground-truth JSON')
    parser.add_argument('--submission', required=True)
    args = parser.parse_args()

    predictions = read_submission(args.submission, args.gt)
    coco_gt = COCO(args.gt)
    coco_dt = load_detections(coco_gt, predictions)
    E = COCOeval_faster(coco_gt, cocoDt=coco_dt, iouType='bbox', print_function=print)
    E.evaluate()
    E.accumulate()
    E.summarize()

    names = ['mAP@[.5:.95]', 'AP@0.50', 'AP@0.75', 'AP (small)', 'AP (medium)',
             'AP (large)', 'AR@1', 'AR@10', 'AR@100', 'AR (small)',
             'AR (medium)', 'AR (large)']
    print('\nheadline metrics:')
    for n, v in zip(names, E.stats):
        print(f'  {n:<14s} {v:.4f}')

    print('\nper-class AP@[.5:.95]:')
    for k, v in per_class_ap(E).items():
        print(f'  {k:<30s} {v:.4f}')
