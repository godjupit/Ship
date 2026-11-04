"""Inference-only feature packing experiments; native PyTorch is the default."""
import os
import torch


def feature_cat(parts):
    if os.environ.get("FEATURE_CAT", "torch") == "torch":
        return torch.cat(parts, dim=1)
    if os.environ["FEATURE_CAT"] != "triton_half":
        raise ValueError("FEATURE_CAT must be torch or triton_half")
    if torch.is_grad_enabled():
        raise ValueError("triton_half feature packing is inference-only")
    if any(t.ndim != 4 or not t.is_cuda for t in parts):
        raise ValueError("triton_half requires CUDA NCHW feature tensors")
    from .feature_pack_triton import pack_half
    return pack_half(parts)
