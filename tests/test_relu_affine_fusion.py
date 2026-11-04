"""Protect separate float32 arithmetic and layout across CUDA Graph replays."""
import pytest
import torch

from engine.misc.relu_affine_triton import relu_affine,FusedReLUAffine
from engine.backbone.hgnetv2 import LearnableAffineBlock

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('layout',['nchw','channels_last','strided'])
def test_matches_original_relu_mul_add_exactly(dtype,layout):
    generator=torch.Generator(device='cuda').manual_seed(123)
    x=torch.randn((2,7,23,19),generator=generator,device='cuda',dtype=dtype)
    if layout=='channels_last':x=x.contiguous(memory_format=torch.channels_last)
    if layout=='strided':x=x[:,:,:,::2]
    scale=torch.tensor([-1.1379],device='cuda');bias=torch.tensor([.08613],device='cuda')
    expected=torch.relu(x)*scale+bias
    with torch.inference_mode():actual=relu_affine(x,scale,bias)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert actual.dtype==torch.float32
    if layout=='channels_last':assert actual.is_contiguous(memory_format=torch.channels_last)


def test_nonfinite_and_zero_semantics():
    x=torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,1.,-1.],device='cuda').view(1,1,1,7)
    scale=torch.tensor([1.37],device='cuda');bias=torch.tensor([-.18],device='cuda')
    with torch.inference_mode():actual=relu_affine(x,scale,bias)
    torch.testing.assert_close(actual,torch.relu(x)*scale+bias,rtol=0,atol=0,equal_nan=True)


def test_graph_replay_reads_current_inputs_and_scalars():
    x=torch.zeros((2,8,16,16),device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    original=LearnableAffineBlock().cuda().eval();fused=FusedReLUAffine(original).eval()
    with torch.inference_mode():
        for _ in range(3):fused(x)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=fused(x)
        x.normal_();original.scale.fill_(1.234);original.bias.fill_(-.765)
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(captured,original(torch.relu(x)),rtol=0,atol=0)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_half_store_keeps_float32_scalar_arithmetic(dtype):
    x=torch.randn((2,11,19,23),device='cuda',dtype=dtype).contiguous(memory_format=torch.channels_last)
    scale=torch.tensor([1.137923],device='cuda');bias=torch.tensor([-.073195],device='cuda')
    expected=(torch.relu(x)*scale+bias).half()
    with torch.inference_mode():actual=relu_affine(x,scale,bias,output_half=True)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert actual.dtype==torch.float16


@pytest.mark.parametrize('half',[False,True])
def test_apply_options_replaces_only_eligible_blocks(monkeypatch,half):
    from engine.backbone.hgnetv2 import ConvBNAct
    from supership.kernel_optimizations import apply_kernel_options
    model=torch.nn.Sequential(ConvBNAct(3,7,3,use_lab=True)).cuda().eval()
    keys=list(model.state_dict())
    for key,value in dict(KERNEL_WEIGHT_HALF='0',KERNEL_AFFINE_HALF='0',KERNEL_LINEAR1X1='0',KERNEL_RELU_AFFINE_FUSE='1',KERNEL_RELU_AFFINE_HALF_OUT=str(int(half))).items():monkeypatch.setenv(key,value)
    changes=apply_kernel_options(model,'fp16')
    assert changes=={'relu_affine_fused':1}
    assert isinstance(model[0].act,torch.nn.Identity)
    assert isinstance(model[0].lab,FusedReLUAffine)
    assert model[0].lab.output_half==half
    assert list(model.state_dict())==keys
