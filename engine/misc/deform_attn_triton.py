"""Inference-only bilinear sampling and weighted reduction, without feature reorders."""
import torch
import triton
import triton.language as tl


@triton.jit
def _level(V, L, A, b, q, head, channel, active,
           H: tl.constexpr, W: tl.constexpr, P: tl.constexpr, O: tl.constexpr,
           VB: tl.constexpr, VH: tl.constexpr, VC: tl.constexpr, VS: tl.constexpr,
           LB: tl.constexpr, LQ: tl.constexpr, LH: tl.constexpr, LP: tl.constexpr, LX: tl.constexpr,
           AB: tl.constexpr, AQ: tl.constexpr, AH: tl.constexpr, AP: tl.constexpr):
    result = tl.full(channel.shape, 0, tl.float32)
    for point in tl.static_range(P):
        loc = b * LB + q * LQ + head * LH + (O + point) * LP
        sx = tl.load(L + loc, active, other=0).to(tl.float32)
        sy = tl.load(L + loc + LX, active, other=0).to(tl.float32)
        # Preserve the float32 grid normalization used by PyTorch align_corners=False.
        gx = 2.0 * sx - 1.0
        gy = 2.0 * sy - 1.0
        x = ((gx + 1.0) * W - 1.0) * 0.5
        y = ((gy + 1.0) * H - 1.0) * 0.5
        x0 = tl.floor(x).to(tl.int32)
        y0 = tl.floor(y).to(tl.int32)
        x1, y1 = x0 + 1, y0 + 1
        base = b * VB + head * VH + channel * VC
        nw = tl.load(V + base + (y0 * W + x0) * VS,
                     active & (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H), other=0).to(tl.float32)
        ne = tl.load(V + base + (y0 * W + x1) * VS,
                     active & (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H), other=0).to(tl.float32)
        sw = tl.load(V + base + (y1 * W + x0) * VS,
                     active & (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H), other=0).to(tl.float32)
        se = tl.load(V + base + (y1 * W + x1) * VS,
                     active & (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H), other=0).to(tl.float32)
        sample = (nw * (x1 - x) * (y1 - y) + ne * (x - x0) * (y1 - y)
                  + sw * (x1 - x) * (y - y0) + se * (x - x0) * (y - y0))
        weight = tl.load(A + b * AB + q * AQ + head * AH + (O + point) * AP,
                         active, other=0).to(tl.float32)
        result += sample * weight
    return result


@triton.jit
def _attention(V0, V1, V2, L, A, OUT,
               N: tl.constexpr, Q: tl.constexpr, HEADS: tl.constexpr, C: tl.constexpr,
               H0: tl.constexpr, W0: tl.constexpr, P0: tl.constexpr,
               H1: tl.constexpr, W1: tl.constexpr, P1: tl.constexpr,
               H2: tl.constexpr, W2: tl.constexpr, P2: tl.constexpr,
               V0B: tl.constexpr, V0H: tl.constexpr, V0C: tl.constexpr, V0S: tl.constexpr,
               V1B: tl.constexpr, V1H: tl.constexpr, V1C: tl.constexpr, V1S: tl.constexpr,
               V2B: tl.constexpr, V2H: tl.constexpr, V2C: tl.constexpr, V2S: tl.constexpr,
               LB: tl.constexpr, LQ: tl.constexpr, LH: tl.constexpr, LP: tl.constexpr, LX: tl.constexpr,
               AB: tl.constexpr, AQ: tl.constexpr, AH: tl.constexpr, AP: tl.constexpr,
               BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = index % C
    head = (index // C) % HEADS
    q = (index // (C * HEADS)) % Q
    b = index // (C * HEADS * Q)
    active = index < N * Q * HEADS * C
    r0 = _level(V0, L, A, b, q, head, channel, active, H0, W0, P0, 0,
                V0B, V0H, V0C, V0S, LB, LQ, LH, LP, LX, AB, AQ, AH, AP)
    r1 = _level(V1, L, A, b, q, head, channel, active, H1, W1, P1, P0,
                V1B, V1H, V1C, V1S, LB, LQ, LH, LP, LX, AB, AQ, AH, AP)
    r2 = _level(V2, L, A, b, q, head, channel, active, H2, W2, P2, P0 + P1,
                V2B, V2H, V2C, V2S, LB, LQ, LH, LP, LX, AB, AQ, AH, AP)
    tl.store(OUT + index, r0 + r1 + r2, active)


def deform_attention(value, spatial_shapes, locations, weights, points):
    if torch.is_grad_enabled() or len(value) != 3 or len(points) != 3:
        raise ValueError('Triton attention requires inference and exactly three feature levels')
    if not all(t.is_cuda for t in [*value, locations, weights]):
        raise ValueError('Triton attention requires CUDA tensors')
    n, heads, channels, _ = value[0].shape
    q = locations.shape[1]
    if (locations.shape != (n, q, heads, sum(points), 2)
            or weights.shape != (n, q, heads, sum(points))):
        raise ValueError('attention locations/weights shape mismatch')
    for v, (h, w) in zip(value, spatial_shapes):
        if v.shape != (n, heads, channels, h * w):
            raise ValueError('value shape mismatch')
    # PyTorch autocast grid_sample promotes the half feature/float grid to float32.
    output = torch.empty((n, q, heads * channels), device=value[0].device, dtype=torch.float32)
    dimensions = [x for (h, w), p in zip(spatial_shapes, points) for x in (h, w, p)]
    strides = [x for v in value for x in v.stride()]
    _attention[(triton.cdiv(output.numel(), 128),)](
        *value, locations, weights, output, n, q, heads, channels,
        *dimensions, *strides, *locations.stride(), *weights.stride(), 128,
        num_warps=4, enable_fp_fusion=False)
    return output
