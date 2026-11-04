"""Fold evaluation BatchNorm into explicitly connected Conv2d weights."""

import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from engine.backbone.common import ConvNormLayer as BackboneConvNormLayer, FrozenBatchNorm2d
from engine.backbone.hgnetv2 import ConvBNAct
from engine.rtv4.hybrid_encoder import ConvNormLayer, ConvNormLayer_fuse


def fuse_model_conv_bn(model):
    """Mutate a loaded eval model; return the names of BN modules removed.

    Only known forward connections and adjacent Sequential entries are folded.
    Padding, activations, branch sums and decoder deployment are left intact.
    The original checkpoint remains unchanged. Call before warmup/graph capture.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Conv-BN fusion requires an evaluation model")
    fused = []

    def fold(parent, conv_name, bn_name, prefix):
        conv, bn = parent._modules[conv_name], parent._modules[bn_name]
        if not isinstance(bn, (nn.BatchNorm2d, FrozenBatchNorm2d)):
            return
        # HGNet's padded convolution ends in Conv2d inside a Sequential.
        container, key = parent, conv_name
        if isinstance(conv, nn.Sequential) and len(conv):
            container = conv
            key = list(conv._modules)[-1]
            conv = conv._modules[key]
        if not isinstance(conv, nn.Conv2d):
            return
        if bn.running_mean is None or bn.running_var is None:
            raise ValueError(f"BatchNorm has no running statistics: {prefix}{bn_name}")
        container._modules[key] = fuse_conv_bn_eval(conv, bn)
        parent._modules[bn_name] = nn.Identity().eval()
        fused.append(prefix + bn_name)

    # Snapshot the hierarchy so replacing modules cannot change iteration order.
    for name, module in list(model.named_modules()):
        prefix = name + "." if name else ""
        if type(module) is ConvBNAct:
            fold(module, "conv", "bn", prefix)
        elif type(module) in (ConvNormLayer, ConvNormLayer_fuse, BackboneConvNormLayer):
            if "conv" in module._modules and "norm" in module._modules:
                fold(module, "conv", "norm", prefix)
        elif type(module) is nn.Sequential:
            keys = list(module._modules)
            for left, right in zip(keys, keys[1:]):
                if isinstance(module._modules[left], nn.Conv2d):
                    fold(module, left, right, prefix)
    return fused
