"""Opt-in inference experiments applied after strict checkpoint loading and BN fusion."""
import os
import torch
from torch import nn
from torch.nn import functional as F


class LinearPointwise(nn.Module):
    """Use NHWC linear algebra for an ungrouped, unit-stride 1x1 convolution."""
    def __init__(self, conv):
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach()[:, :, 0, 0].contiguous(), requires_grad=False)
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach(), requires_grad=False)

    def forward(self, x):
        return F.linear(x.permute(0, 2, 3, 1), self.weight, self.bias).permute(0, 3, 1, 2)


def apply_kernel_options(model, precision):
    options = {key: os.environ.get(key, "0") == "1" for key in
               ["KERNEL_WEIGHT_HALF", "KERNEL_AFFINE_HALF", "KERNEL_LINEAR1X1", "KERNEL_RELU_AFFINE_FUSE", "KERNEL_RELU_AFFINE_HALF_OUT"]}
    if options["KERNEL_RELU_AFFINE_HALF_OUT"] and not options["KERNEL_RELU_AFFINE_FUSE"]:
        raise ValueError("half output requires ReLU affine fusion")
    if not any(options.values()):
        return {}
    if model.training or precision != "fp16":
        raise ValueError("kernel experiments require evaluation mode and fp16 autocast")
    changes = {}
    if options["KERNEL_WEIGHT_HALF"]:
        modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        for module in modules:
            module.half()
        changes["weight_half"] = len(modules)
    if options["KERNEL_AFFINE_HALF"]:
        from engine.backbone.hgnetv2 import LearnableAffineBlock
        modules = [m for m in model.modules() if isinstance(m, LearnableAffineBlock)]
        for module in modules:
            module.half()
        changes["affine_half"] = len(modules)
    if options["KERNEL_LINEAR1X1"]:
        count = 0
        for parent in list(model.modules()):
            for name, child in list(parent.named_children()):
                if (isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1)
                        and child.stride == (1, 1) and child.padding == (0, 0)
                        and child.groups == 1 and child.padding_mode == "zeros"):
                    parent._modules[name] = LinearPointwise(child).eval()
                    count += 1
        changes["linear_1x1"] = count
    if options["KERNEL_RELU_AFFINE_FUSE"]:
        from engine.backbone.hgnetv2 import ConvBNAct, LearnableAffineBlock
        from engine.misc.relu_affine_triton import FusedReLUAffine
        count=0
        for module in list(model.modules()):
            if isinstance(module,ConvBNAct) and isinstance(module.act,nn.ReLU) and isinstance(module.lab,LearnableAffineBlock):
                if module.lab.scale.dtype!=torch.float32 or module.lab.bias.dtype!=torch.float32:
                    raise ValueError("ReLU affine fusion requires the original float32 affine scalars")
                module.act=nn.Identity()
                module.lab=FusedReLUAffine(module.lab,options["KERNEL_RELU_AFFINE_HALF_OUT"]).eval()
                count+=1
        changes["relu_affine_fused"]=count
    return changes
