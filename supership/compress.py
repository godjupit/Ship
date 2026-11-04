"""Structurally prune decoder depth and recover with supervised distillation."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert

from supership.generate_submission import build_runtime, create_loader


def distillation_loss(student, teacher):
    """Match prediction sets instead of assuming identical query ordering."""
    losses = []
    for logits, boxes, t_logits, t_boxes in zip(
        student["pred_logits"], student["pred_boxes"],
        teacher["pred_logits"], teacher["pred_boxes"],
    ):
        confidence = t_logits.sigmoid().amax(-1)
        keep = confidence.topk(min(32, confidence.numel())).indices
        keep = keep[confidence[keep] >= 0.3]
        if keep.numel() == 0:
            continue
        cost = torch.cdist(boxes.detach(), t_boxes[keep], p=1)
        cost += torch.cdist(logits.detach().sigmoid(), t_logits[keep].sigmoid(), p=1)
        source, target = linear_sum_assignment(cost.cpu().numpy())
        source = torch.as_tensor(source, device=boxes.device)
        target = keep[torch.as_tensor(target, device=boxes.device)]
        losses.append(5 * F.l1_loss(boxes[source], t_boxes[target])
                      + F.mse_loss(logits[source].sigmoid(), t_logits[target].sigmoid()))
    return torch.stack(losses).mean() if losses else student["pred_logits"].sum() * 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dfine/dfine_hgnetv2_s_supership.yml")
    parser.add_argument("--weights", default="weights/baseline.pth")
    parser.add_argument("--output-dir", default="output/compression")
    parser.add_argument("--student-layers", type=int, default=2, help="2 prunes one of the baseline's 3 decoder layers")
    parser.add_argument("--steps", type=int, default=20, help="total training steps; 0=prune only, -1=all epochs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kd-weight", type=float, default=1.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.student_layers <= 3 or args.steps < -1 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0 or args.kd_weight < 0:
        parser.error("Invalid training parameters")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _, teacher, _ = build_runtime(args.config, args.weights, args.device)
    teacher.requires_grad_(False).eval()
    from engine.core import YAMLConfig

    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    overrides = {"DFINETransformer": {"num_layers": args.student_layers, "eval_idx": -1}}
    cfg.yaml_cfg["DFINETransformer"].update(overrides["DFINETransformer"])
    # Use supervised classification/L1/GIoU losses; cross-model KD is added below.
    cfg.yaml_cfg["RTv4Criterion"]["losses"] = ["vfl", "boxes"]
    student = cfg.model.to(args.device)
    target_state = student.state_dict()
    source_state = teacher.state_dict()
    compatible = {k: v for k, v in source_state.items()
                  if k in target_state and v.shape == target_state[k].shape}
    missing = set(target_state) - set(compatible)
    if missing:
        raise RuntimeError(f"Cannot initialize pruned model: {sorted(missing)}")
    student.load_state_dict(compatible, strict=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parameters = vars(args) | {"training_split": "train", "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
                              "student_parameters": sum(p.numel() for p in student.parameters())}
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2))

    def save(name, steps, optimizer=None):
        checkpoint = {"model": {k: v.detach().cpu() for k, v in student.state_dict().items()},
                      "model_overrides": overrides, "compression": parameters | {"completed_steps": steps}}
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        torch.save(checkpoint, output / name)

    save("pruned_initial.pth", 0)
    print(f"Parameters: {parameters['teacher_parameters']} -> {parameters['student_parameters']}", flush=True)
    if args.steps == 0:
        return
    criterion = cfg.criterion.to(args.device)
    cfg.yaml_cfg["val_dataloader"].update(num_workers=args.workers, shuffle=True)
    loader = create_loader(cfg, "data/SuperShip/train/images",
                           "data/SuperShip/train/annotations/instances_train.json", args.batch_size)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    step = 0
    with (output / "training.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            student.train()
            # Preserve pretrained running statistics during short recovery runs.
            for module in student.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            for samples, targets in loader:
                samples = samples.to(args.device)
                targets = [{k: v.to(args.device) for k, v in target.items()} for target in targets]
                for target in targets:
                    target["boxes"] = box_convert(target["boxes"], "xyxy", "cxcywh") / 640.
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    reference = teacher(samples)
                prediction = student(samples, targets=targets)
                losses = criterion(prediction, targets, epoch=epoch, step=step, global_step=step)
                supervised = sum(losses.values())
                kd = distillation_loss(prediction, reference)
                loss = supervised + args.kd_weight * kd
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError(f"Nonfinite gradients at step {step}")
                optimizer.step()
                step += 1
                record = dict(step=step, epoch=epoch, loss=loss.item(), supervised=supervised.item(), kd=kd.item())
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(record, flush=True)
                if args.steps > 0 and step >= args.steps:
                    break
            save("distilled.pth", step, optimizer)
            if args.steps > 0 and step >= args.steps:
                break
    print(f"Saved {output / 'distilled.pth'} after {step} training steps", flush=True)


if __name__ == "__main__":
    main()
