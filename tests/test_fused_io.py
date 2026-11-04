"""Protect exact arithmetic and static-buffer updates for fused IO kernels."""
import pytest
import torch
import torchvision

from supership.fused_io import normalized_half, packed_detections
from engine.misc.feature_pack_triton import pack_half

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def native_output(logits, boxes, sizes, k):
    coords = torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")
    coords *= sizes.repeat(1, 2).unsqueeze(1)
    scores, index = logits.sigmoid().flatten(1).topk(k, dim=-1)
    selected = coords.gather(1, (index // logits.shape[-1]).unsqueeze(-1).repeat(1, 1, 4))
    return torch.cat((selected, scores.unsqueeze(-1), (index % logits.shape[-1]).unsqueeze(-1)), -1)


def test_input_all_byte_values_and_graph_replay():
    x = torch.arange(256, device="cuda").to(torch.uint8).repeat(3).reshape(1, 16, 16, 3)
    expected = x.permute(0, 3, 1, 2).float().mul_(1 / 255).half()
    with torch.inference_mode():
        torch.testing.assert_close(normalized_half(x), expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = normalized_half(x)
        x.bitwise_not_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, x.permute(0, 3, 1, 2).float().mul_(1 / 255).half(), rtol=0, atol=0)
        assert output.is_contiguous(memory_format=torch.channels_last)


def test_bgr_conversion_and_normalization_are_exact():
    x = torch.randint(0, 256, (2, 37, 29, 3), device="cuda", dtype=torch.uint8)
    with torch.inference_mode():
        actual = normalized_half(x, bgr=True)
    expected = x.flip(-1).permute(0, 3, 1, 2).float().mul_(1 / 255).half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("b,q,k", [(1, 300, 300), (2, 150, 200), (3, 100, 100)])
def test_detection_geometry_scores_labels_exact(b, q, k):
    torch.manual_seed(92)
    logits = torch.randn(b, q, 3, device="cuda")
    boxes = torch.randn(b, q, 4, device="cuda")
    sizes = torch.tensor([[701., 1129.]], device="cuda").repeat(b, 1)
    with torch.inference_mode():
        actual = packed_detections(dict(pred_logits=logits, pred_boxes=boxes), sizes, k)
    torch.testing.assert_close(actual, native_output(logits, boxes, sizes, k), rtol=0, atol=0)


def test_detection_graph_reads_updated_boxes_logits_sizes():
    logits = torch.randn(2, 300, 3, device="cuda")
    boxes = torch.rand(2, 300, 4, device="cuda")
    sizes = torch.ones(2, 2, device="cuda")
    with torch.inference_mode():
        packed_detections(dict(pred_logits=logits, pred_boxes=boxes), sizes)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = packed_detections(dict(pred_logits=logits, pred_boxes=boxes), sizes)
        logits.normal_(); boxes.uniform_(); sizes.fill_(1024.)
        graph.replay(); torch.cuda.synchronize()
        torch.testing.assert_close(output, native_output(logits, boxes, sizes, 300), rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["channels_last", "nchw", "strided"])
@pytest.mark.parametrize("block", [256, 1024])
def test_segment_pack_matches_native_for_unequal_channels(monkeypatch, layout, block):
    monkeypatch.setenv("FEATURE_PACK_IMPL", "segmented")
    monkeypatch.setenv("FEATURE_PACK_BLOCK", str(block))
    parts = []
    for c in [3, 8, 19, 7]:
        x = torch.randn(2, c, 11, 18, device="cuda")
        if layout == "channels_last":
            x = x.contiguous(memory_format=torch.channels_last)
        elif layout == "strided":
            x = x[..., ::2]
        parts.append(x)
    with torch.inference_mode():
        output = pack_half(parts)
    torch.testing.assert_close(output, torch.cat(parts, 1).half(), rtol=0, atol=0)
    assert output.is_contiguous(memory_format=torch.channels_last)


def test_segment_pack_replay_updates_every_segment(monkeypatch):
    monkeypatch.setenv("FEATURE_PACK_IMPL", "segmented")
    parts = [torch.zeros(1, c, 9, 7, device="cuda", dtype=torch.float16) for c in [4, 7, 3]]
    with torch.inference_mode():
        pack_half(parts)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = pack_half(parts)
        for i, x in enumerate(parts):
            x.fill_(i + 1)
        graph.replay(); torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.cat(parts, 1), rtol=0, atol=0)
