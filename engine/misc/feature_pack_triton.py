"""Concatenate feature channels and cast once into an NHWC FP16 destination."""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _segment(P, OUT, index, n, h, w, channel, TOTAL: tl.constexpr,
             C: tl.constexpr, O: tl.constexpr, SN: tl.constexpr,
             SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr):
    if C > 0:
        valid = (index < TOTAL) & (channel >= O) & (channel < O + C)
        source = n * SN + (channel - O) * SC + h * SH + w * SW
        value = tl.load(P + source, mask=valid, other=0)
        tl.store(OUT + index, value, mask=valid)


@triton.jit
def _pack(P0, P1, P2, P3, P4, P5, P6, P7, OUT,
          H: tl.constexpr, W: tl.constexpr, CO: tl.constexpr,
          TOTAL: tl.constexpr, BLOCK: tl.constexpr,
          C0: tl.constexpr, O0: tl.constexpr, SN0: tl.constexpr, SC0: tl.constexpr, SH0: tl.constexpr, SW0: tl.constexpr,
          C1: tl.constexpr, O1: tl.constexpr, SN1: tl.constexpr, SC1: tl.constexpr, SH1: tl.constexpr, SW1: tl.constexpr,
          C2: tl.constexpr, O2: tl.constexpr, SN2: tl.constexpr, SC2: tl.constexpr, SH2: tl.constexpr, SW2: tl.constexpr,
          C3: tl.constexpr, O3: tl.constexpr, SN3: tl.constexpr, SC3: tl.constexpr, SH3: tl.constexpr, SW3: tl.constexpr,
          C4: tl.constexpr, O4: tl.constexpr, SN4: tl.constexpr, SC4: tl.constexpr, SH4: tl.constexpr, SW4: tl.constexpr,
          C5: tl.constexpr, O5: tl.constexpr, SN5: tl.constexpr, SC5: tl.constexpr, SH5: tl.constexpr, SW5: tl.constexpr,
          C6: tl.constexpr, O6: tl.constexpr, SN6: tl.constexpr, SC6: tl.constexpr, SH6: tl.constexpr, SW6: tl.constexpr,
          C7: tl.constexpr, O7: tl.constexpr, SN7: tl.constexpr, SC7: tl.constexpr, SH7: tl.constexpr, SW7: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, channel = index // CO, index % CO
    n, h, w = row // (H * W), (row // W) % H, row % W
    _segment(P0, OUT, index, n, h, w, channel, TOTAL, C0, O0, SN0, SC0, SH0, SW0)
    _segment(P1, OUT, index, n, h, w, channel, TOTAL, C1, O1, SN1, SC1, SH1, SW1)
    _segment(P2, OUT, index, n, h, w, channel, TOTAL, C2, O2, SN2, SC2, SH2, SW2)
    _segment(P3, OUT, index, n, h, w, channel, TOTAL, C3, O3, SN3, SC3, SH3, SW3)
    _segment(P4, OUT, index, n, h, w, channel, TOTAL, C4, O4, SN4, SC4, SH4, SW4)
    _segment(P5, OUT, index, n, h, w, channel, TOTAL, C5, O5, SN5, SC5, SH5, SW5)
    _segment(P6, OUT, index, n, h, w, channel, TOTAL, C6, O6, SN6, SC6, SH6, SW6)
    _segment(P7, OUT, index, n, h, w, channel, TOTAL, C7, O7, SN7, SC7, SH7, SW7)



@triton.jit
def _scatter_segment(P, OUT, H: tl.constexpr, W: tl.constexpr, CO: tl.constexpr,
                     TOTAL: tl.constexpr, BLOCK: tl.constexpr, PART: tl.constexpr,
                     C: tl.constexpr, O: tl.constexpr, SN: tl.constexpr,
                     SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr):
    if C > 0:
        if tl.program_id(1) == PART:
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            valid = i < (TOTAL // CO) * C
            row, channel = i // C, i % C
            n, h, w = row // (H * W), (row // W) % H, row % W
            value = tl.load(P + n * SN + channel * SC + h * SH + w * SW, valid, other=0)
            tl.store(OUT + row * CO + O + channel, value, valid)


@triton.jit
def _pack_segmented(P0, P1, P2, P3, P4, P5, P6, P7, OUT,
                    H: tl.constexpr, W: tl.constexpr, CO: tl.constexpr,
                    TOTAL: tl.constexpr, BLOCK: tl.constexpr,
          C0: tl.constexpr, O0: tl.constexpr, SN0: tl.constexpr, SC0: tl.constexpr, SH0: tl.constexpr, SW0: tl.constexpr,
          C1: tl.constexpr, O1: tl.constexpr, SN1: tl.constexpr, SC1: tl.constexpr, SH1: tl.constexpr, SW1: tl.constexpr,
          C2: tl.constexpr, O2: tl.constexpr, SN2: tl.constexpr, SC2: tl.constexpr, SH2: tl.constexpr, SW2: tl.constexpr,
          C3: tl.constexpr, O3: tl.constexpr, SN3: tl.constexpr, SC3: tl.constexpr, SH3: tl.constexpr, SW3: tl.constexpr,
          C4: tl.constexpr, O4: tl.constexpr, SN4: tl.constexpr, SC4: tl.constexpr, SH4: tl.constexpr, SW4: tl.constexpr,
          C5: tl.constexpr, O5: tl.constexpr, SN5: tl.constexpr, SC5: tl.constexpr, SH5: tl.constexpr, SW5: tl.constexpr,
          C6: tl.constexpr, O6: tl.constexpr, SN6: tl.constexpr, SC6: tl.constexpr, SH6: tl.constexpr, SW6: tl.constexpr,
          C7: tl.constexpr, O7: tl.constexpr, SN7: tl.constexpr, SC7: tl.constexpr, SH7: tl.constexpr, SW7: tl.constexpr):
    _scatter_segment(P0, OUT, H, W, CO, TOTAL, BLOCK, 0, C0, O0, SN0, SC0, SH0, SW0)
    _scatter_segment(P1, OUT, H, W, CO, TOTAL, BLOCK, 1, C1, O1, SN1, SC1, SH1, SW1)
    _scatter_segment(P2, OUT, H, W, CO, TOTAL, BLOCK, 2, C2, O2, SN2, SC2, SH2, SW2)
    _scatter_segment(P3, OUT, H, W, CO, TOTAL, BLOCK, 3, C3, O3, SN3, SC3, SH3, SW3)
    _scatter_segment(P4, OUT, H, W, CO, TOTAL, BLOCK, 4, C4, O4, SN4, SC4, SH4, SW4)
    _scatter_segment(P5, OUT, H, W, CO, TOTAL, BLOCK, 5, C5, O5, SN5, SC5, SH5, SW5)
    _scatter_segment(P6, OUT, H, W, CO, TOTAL, BLOCK, 6, C6, O6, SN6, SC6, SH6, SW6)
    _scatter_segment(P7, OUT, H, W, CO, TOTAL, BLOCK, 7, C7, O7, SN7, SC7, SH7, SW7)

def pack_half(parts):
    if not 1 <= len(parts) <= 8:
        raise ValueError("feature packing supports one to eight tensors")
    n, _, h, w = parts[0].shape
    if any((t.shape[0], t.shape[2], t.shape[3]) != (n, h, w) for t in parts):
        raise ValueError("feature dimensions must match")
    channels = [t.shape[1] for t in parts]
    output = torch.empty((n, h, w, sum(channels)), device=parts[0].device, dtype=torch.float16)
    pointers = list(parts) + [parts[-1]] * (8 - len(parts))
    metadata, offset = [], 0
    for i in range(8):
        if i < len(parts):
            metadata.extend([channels[i], offset, *parts[i].stride()])
            offset += channels[i]
        else:
            metadata.extend([0, 0, 0, 0, 0, 0])
    total = output.numel()
    block = int(os.environ.get("FEATURE_PACK_BLOCK", "256"))
    if block not in (256, 512, 1024, 2048):
        raise ValueError("unsupported feature pack block size")
    if os.environ.get("FEATURE_PACK_IMPL", "masked") == "segmented":
        grid = (triton.cdiv(n * h * w * max(channels), block), len(parts))
        _pack_segmented[grid](*pointers, output, h, w, sum(channels), total, block, *metadata)
    else:
        _pack[(triton.cdiv(total, block),)](*pointers, output, h, w, sum(channels),
                                         total, block, *metadata)
    return output.permute(0, 3, 1, 2)
