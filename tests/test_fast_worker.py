"""Check the data contract of the threaded preprocessing path."""

import numpy as np
from PIL import Image
import pytest
from torchvision.transforms import v2

from supership.fast_worker import Runtime, ordered_items


def make_image(tmp_path):
    pixels = np.arange(37 * 23 * 3, dtype=np.uint8).reshape(23, 37, 3)
    path = tmp_path / "sample.png"
    Image.fromarray(pixels).save(path)
    return path, {"id": 17, "file_name": path.name, "width": 37, "height": 23}


def test_pil_pipeline_preserves_official_resize(tmp_path):
    path, item = make_image(tmp_path)
    runtime = Runtime.__new__(Runtime)
    runtime.decoder = "pil"
    result = np.empty((640, 640, 3), dtype=np.uint8)
    assert runtime.load_image(item, tmp_path, result) == (37, 23)
    with Image.open(path) as image:
        expected = np.asarray(v2.Resize([640, 640])(image.convert("RGB")))
    np.testing.assert_array_equal(result, expected)


def test_image_dimensions_checked_against_metadata(tmp_path):
    _, item = make_image(tmp_path)
    item["width"] += 1
    runtime = Runtime.__new__(Runtime)
    runtime.decoder = "pil"
    with pytest.raises(ValueError, match="metadata dimensions"):
        runtime.load_image(item, tmp_path, np.empty((640, 640, 3), dtype=np.uint8))


def test_opencv_pipeline_keeps_rgb_order(tmp_path):
    import cv2

    path, item = make_image(tmp_path)
    runtime = Runtime.__new__(Runtime)
    runtime.decoder = "opencv"
    runtime.reduction = 1
    result = np.empty((640, 640, 3), dtype=np.uint8)
    runtime.load_image(item, tmp_path, result)
    with Image.open(path) as image:
        expected = cv2.resize(np.asarray(image), (640, 640), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('reduction', [2, 4, 8])
def test_reduced_jpeg_preserves_original_coordinate_scale(tmp_path, reduction):
    path = tmp_path / "odd.jpg"
    Image.new("RGB", (47, 35), (50, 100, 150)).save(path)
    item = {"id": 3, "file_name": path.name, "width": 47, "height": 35}
    runtime = Runtime.__new__(Runtime)
    runtime.decoder = "opencv"
    runtime.reduction = reduction
    result = np.empty((640, 640, 3), dtype=np.uint8)
    assert runtime.load_image(item, tmp_path, result) == (47, 35)
    assert result.shape == (640, 640, 3)


@pytest.mark.parametrize('mode', ['metadata', 'small-first', 'large-first', 'balanced'])
@pytest.mark.parametrize('count', [0, 1, 7, 8, 9, 25])
def test_ordering_preserves_each_input_and_tail_capacity(tmp_path, mode, count):
    items = []
    for index in range(count):
        name = f'{index}.jpg'
        (tmp_path / name).write_bytes(b'x' * ((index * 17) % 31 + 1))
        items.append(dict(id=index, file_name=name))
    result = ordered_items(items, tmp_path, 8, mode)
    assert sorted(x['id'] for x in result) == list(range(count))
    assert ordered_items(items, tmp_path, 8, mode) == result
    if mode in ('small-first', 'large-first'):
        sizes = [(tmp_path / x['file_name']).stat().st_size for x in result]
        assert sizes == sorted(sizes, reverse=mode == 'large-first')
