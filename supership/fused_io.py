"""Opt-in CUDA input conversion and packed inference detections."""
import torch
import triton
import triton.language as tl


@triton.jit
def _normalize(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, BGR: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = (i // 3) * 3 + 2 - i % 3 if BGR else i
    x = tl.load(X + source, i < N, other=0).to(tl.float32)
    tl.store(Y + i, x * (1.0 / 255.0), i < N)


def normalized_half(images, bgr=False):
    if not images.is_cuda or images.dtype != torch.uint8 or not images.is_contiguous():
        raise ValueError("normalization requires contiguous CUDA uint8 NHWC")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("normalization requires three-channel NHWC")
    output = torch.empty_like(images, dtype=torch.float16)
    n = images.numel()
    if n:
        _normalize[(triton.cdiv(n, 1024),)](images, output, n, 1024, bgr)
    return output.permute(0, 3, 1, 2)


@triton.jit
def _detections(BOXES, SCORES, INDEX, SIZES, OUT, K: tl.constexpr,
                Q: tl.constexpr, C: tl.constexpr, TOTAL: tl.constexpr,
                BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < TOTAL
    batch = i // K
    index = tl.load(INDEX + i, valid, other=0)
    query = index // C
    label = index % C
    pointer = BOXES + (batch * Q + query) * 4
    cx = tl.load(pointer, valid, other=0)
    cy = tl.load(pointer + 1, valid, other=0)
    w = tl.load(pointer + 2, valid, other=0)
    h = tl.load(pointer + 3, valid, other=0)
    iw = tl.load(SIZES + batch * 2, valid, other=1)
    ih = tl.load(SIZES + batch * 2 + 1, valid, other=1)
    score = tl.load(SCORES + i, valid, other=0).to(tl.float32)
    tl.store(OUT + i * 6, (cx - w * 0.5) * iw, valid)
    tl.store(OUT + i * 6 + 1, (cy - h * 0.5) * ih, valid)
    tl.store(OUT + i * 6 + 2, (cx + w * 0.5) * iw, valid)
    tl.store(OUT + i * 6 + 3, (cy + h * 0.5) * ih, valid)
    tl.store(OUT + i * 6 + 4, score, valid)
    tl.store(OUT + i * 6 + 5, label.to(tl.float32), valid)


def packed_detections(outputs, sizes, top_queries=300):
    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    if boxes.dtype != torch.float32 or sizes.dtype != torch.float32:
        raise ValueError("packed detection geometry requires float32 boxes and sizes")
    if not (boxes.is_contiguous() and sizes.is_contiguous()):
        raise ValueError("packed detection geometry requires contiguous tensors")
    b, q, c = logits.shape
    if tuple(boxes.shape) != (b, q, 4) or tuple(sizes.shape) != (b, 2):
        raise ValueError("inconsistent packed detection dimensions")
    scores, index = logits.sigmoid().flatten(1).topk(top_queries, dim=-1)
    output = torch.empty((b, top_queries, 6), device=boxes.device, dtype=torch.float32)
    total = b * top_queries
    _detections[(triton.cdiv(total, 256),)](
        boxes, scores, index, sizes, output, top_queries, q, c, total, 256,
        enable_fp_fusion=False)
    return output
