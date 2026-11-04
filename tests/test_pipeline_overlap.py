"""Protect RGB values and pinned ring ownership across async graph replays."""
import json

import numpy as np
from PIL import Image
import pytest
import torch

from supership import fast_worker


@pytest.mark.parametrize('side', [320, 576, 640])
@pytest.mark.parametrize('reduction', [1, 2])
@pytest.mark.parametrize('copy_mode', ['cvtcolor', 'direct'])
def test_copy_modes_are_pixel_exact(tmp_path, reduction, copy_mode, side):
    pixels = np.random.default_rng(42).integers(0, 256, (73, 95, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / 'sample.jpg')
    item = dict(id=1, file_name='sample.jpg', width=95, height=73)
    runtime = fast_worker.Runtime.__new__(fast_worker.Runtime)
    runtime.decoder = 'opencv'
    runtime.reduction = reduction
    expected = np.empty((side, side, 3), dtype=np.uint8)
    actual = np.empty_like(expected)
    runtime.copy_mode = 'strided'
    runtime.load_image(item, tmp_path, expected)
    runtime.copy_mode = copy_mode
    assert runtime.load_image(item, tmp_path, actual) == (95, 73)
    np.testing.assert_array_equal(actual, expected)


class SyntheticModel(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(2, 3))


class SyntheticPost(torch.nn.Module):
    def deploy(self):
        return self

    def forward(self, output, sizes):
        n = output.shape[0]
        boxes = torch.cat((sizes * 0.1, sizes * 0.9), dim=1).unsqueeze(1)
        return torch.zeros((n, 1), device=output.device), boxes, output[:, :1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('backend', ['torch', 'cuda_graph'])
@pytest.mark.parametrize('transport', ['0', '1'])
def test_async_ring_matches_sync_with_tail_and_reuse(tmp_path, monkeypatch, depth, backend, transport):
    monkeypatch.setattr(fast_worker, 'build_runtime',
                        lambda *_: (None, SyntheticModel().cuda(), SyntheticPost().cuda()))
    for key, value in dict(BATCH_SIZE='2', NUM_WORKERS='2', PIPELINE_DEPTH=str(depth),
                           PRECISION='fp32', BACKEND=backend, IMAGE_DECODER='opencv',
                           JPEG_REDUCTION='1', SCORE_THRESHOLD='0', DECODER_LAYERS='0',
                           OPENCV_COPY_MODE='cvtcolor', VECTORIZED_RESULTS='1').items():
        monkeypatch.setenv(key, value)
    items = []
    for index in range(11):
        width, height = 47 + index, 35 + index
        Image.new('RGB', (width, height), (index * 20, 100, 150)).save(tmp_path / f'{index}.png')
        items.append(dict(id=index, file_name=f'{index}.png', width=width, height=height))
    meta = tmp_path / 'meta.json'
    meta.write_text(json.dumps(dict(images=items)))
    results = []
    for enabled in ['0', '1']:
        monkeypatch.setenv('ASYNC_OUTPUT', enabled)
        monkeypatch.setenv('PINNED_SIZES', transport if enabled == '1' else '0')
        monkeypatch.setenv('REUSE_EVENTS', transport if enabled == '1' else '0')
        runtime = fast_worker.Runtime()
        output = tmp_path / f'result-{enabled}.json'
        assert runtime.run(dict(image_dir=str(tmp_path), meta_file=str(meta), output_file=str(output))) == (11, 11)
        results.append(output.read_bytes())
    assert results[0] == results[1]
