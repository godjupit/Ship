"""Keep byte-identical formatting and pinned-output ownership with threaded JSON."""
import json
import numpy as np
import orjson
from PIL import Image
import pytest
import torch

from supership.fast_worker import Runtime, bulk_results, serialized_batch
from supership import fast_worker


def test_serialized_chunks_match_whole_output_with_empty_middle():
    items = [dict(id=i) for i in range(5)]
    output = np.random.default_rng(71).uniform(0, 1, (5, 3, 6)).astype(np.float32)
    output[:, :, :2] *= 100
    output[:, :, 2:4] = output[:, :, :2] + output[:, :, 2:4] * 100
    output[:, :, 5] = np.arange(3)
    output[2, :, 4] = 0
    chunks = [serialized_batch(items[:2], output[:2], .05),
              serialized_batch(items[2:3], output[2:3], .05),
              serialized_batch(items[3:], output[3:], .05)]
    actual = b"[" + b",".join(b for _, b in chunks if b) + b"]"
    expected = bulk_results(items, [output], .05)
    assert sum(n for n, _ in chunks) == len(expected)
    assert actual == orjson.dumps(expected)


class Model(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(2, 3))


class Post(torch.nn.Module):
    def deploy(self):
        return self

    def forward(self, values, sizes):
        boxes = torch.cat((sizes * .1, sizes * .9), 1).unsqueeze(1)
        return values[:, :1].long() * 0, boxes, values[:, :1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("asynchronous", ["0", "1"])
def test_threaded_output_matches_bulk_across_ring_tail_and_empty(tmp_path, monkeypatch, depth, asynchronous):
    monkeypatch.setattr(fast_worker, "build_runtime", lambda *_: (None, Model().cuda(), Post().cuda()))
    for key, value in dict(INPUT_SIZE="640", BATCH_SIZE="2", NUM_WORKERS="2", PIPELINE_DEPTH=str(depth),
                           PRECISION="fp32", BACKEND="cuda_graph", IMAGE_DECODER="opencv", JPEG_REDUCTION="1",
                           SCORE_THRESHOLD="0.05", DECODER_LAYERS="0", VECTORIZED_RESULTS="1", BULK_RESULTS="1",
                           ASYNC_OUTPUT=asynchronous, EARLY_INPUT_RECYCLE="1").items():
        monkeypatch.setenv(key, value)
    items = []
    for i in range(11):
        Image.new("RGB", (37, 29), (i * 20, 50, 70)).save(tmp_path / f"{i}.png")
        items.append(dict(id=i, file_name=f"{i}.png", width=37, height=29))
    metadata = tmp_path / "meta.json"
    outputs = []
    for threaded in ["0", "1"]:
        monkeypatch.setenv("THREADED_JSON", threaded)
        for name, images in [("tail", items), ("empty", [])]:
            metadata.write_text(json.dumps(dict(images=images)))
            runtime = Runtime()
            output = tmp_path / f"{threaded}-{name}.json"
            _, count = runtime.run(dict(image_dir=str(tmp_path), meta_file=str(metadata), output_file=str(output)))
            assert count == len(images)
            outputs.append(output.read_bytes())
    assert outputs[:2] == outputs[2:]
