"""NVTX instrumentation using the production pinned-memory input transfers."""
import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from supership.generate_submission import build_runtime, create_loader, read_worker_request, READY_MARKER, DevicePrefetcher, batch_detections_to_cpu

BATCH = ContextVar('batch_id', default=None)
IMAGE = ContextVar('image_id', default=None)


@contextmanager
def mark(stage, batch=None, image=None):
    batch = BATCH.get() if batch is None else batch
    image = IMAGE.get() if image is None else image
    name = stage
    if batch is not None: name += f'/batch={batch:04d}'
    if image is not None: name += f'/image={image}'
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class TracedTransform:
    def __init__(self, original): self.original = original
    def __call__(self, *args, **kwargs):
        with mark('cpu_preprocess'):
            return self.original(*args, **kwargs)


class TracedCollate:
    def __init__(self, original, batch_by_image):
        self.original, self.batch_by_image = original, batch_by_image
    def __call__(self, items):
        image_id = int(items[0][1]['image_id'].item())
        with mark('batch_assemble', batch=self.batch_by_image[image_id]):
            return self.original(items)


def instrument_loader(loader, batch_size):
    from engine.data.dataset.coco_dataset import CocoDetection
    original_getitem = CocoDetection.__getitem__
    original_load_image = CocoDetection._load_image

    def getitem(dataset, index):
        bt = BATCH.set(index // batch_size)
        im = IMAGE.set(int(dataset.ids[index]))
        try:
            with mark('cpu_sample_total'):
                return original_getitem(dataset, index)
        finally:
            IMAGE.reset(im)
            BATCH.reset(bt)

    def load_image(dataset, image_id):
        # Keep PIL's original lazy I/O + JPEG decode + RGB conversion intact.
        with mark('jpeg_read_decode'):
            return original_load_image(dataset, image_id)

    CocoDetection.__getitem__ = getitem
    CocoDetection._load_image = load_image
    loader.dataset._transforms = TracedTransform(loader.dataset._transforms)
    mapping = {int(image_id): i // batch_size for i, image_id in enumerate(loader.dataset.ids)}
    loader.collate_fn = TracedCollate(loader.collate_fn, mapping)
    return loader


def traced_batches(loader):
    with mark('loader_start_workers'):
        iterator = iter(loader)
    for batch_id in range(len(loader)):
        with mark('loader_wait', batch=batch_id):
            batch = next(iterator)
        yield batch_id, batch
    # Match the original for-loop's StopIteration/worker shutdown before writing.
    with mark('loader_finish_workers'):
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError('DataLoader yielded more batches than len(loader)')


def instrument_graph_replay():
    original = torch.cuda.CUDAGraph.replay
    def replay(graph):
        with mark('cuda_graph_replay'):
            return original(graph)
    torch.cuda.CUDAGraph.replay = replay


def run_inference(cfg, model, postprocessor, device, batch_size, image_dir, meta_file, output_file):
    with mark('loader_create'):
        loader = instrument_loader(create_loader(cfg, image_dir, meta_file, batch_size), batch_size)
    results = []
    batches = (batch for _, batch in traced_batches(loader))
    section = lambda name, **kwargs: mark(name)
    with torch.inference_mode(), DevicePrefetcher(batches, device, section) as prefetcher:
        for batch_id, (samples, targets, original_sizes) in enumerate(prefetcher):
            token = BATCH.set(batch_id)
            try:
                with mark('batch_consume_total'):
                    with mark('gpu_forward_eager_submit'):
                        outputs = model(samples)
                    with mark('gpu_postprocess_submit'):
                        detections = postprocessor(outputs, original_sizes)
                    prefetcher.preload_next()
                    # 保留 d2h 阶段名，现按整个 batch 标记回传。
                    with mark('d2h'):
                        cpu_detections = batch_detections_to_cpu(detections)
                    for target, detection in zip(targets, cpu_detections):
                        image_id = target['image_id'].item()
                        boxes = detection['boxes']
                        scores = detection['scores']
                        labels = detection['labels']
                        with mark('results_format', image=image_id):
                            for index, score in enumerate(scores):
                                x1, y1, x2, y2 = boxes[index]
                                width = round(x2-x1, 2)
                                height = round(y2-y1, 2)
                                results.append(dict(
                                    image_id=image_id, category_id=int(labels[index]),
                                    bbox=[round(x1, 2), round(y1, 2), width, height],
                                    score=round(score, 5),
                                ))
            finally:
                BATCH.reset(token)
    if str(device).startswith('cuda'):
        with mark('existing_final_cuda_sync'):
            torch.cuda.synchronize(device)  # Present in the original worker.
    with mark('output_write'):
        output_path = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as output:
            json.dump(results, output, separators=(',', ':'), allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--config', default='configs/dfine/dfine_hgnetv2_s_supership.yml')
    parser.add_argument('--weights', default='weights/baseline.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    os.chdir(ROOT)
    instrument_graph_replay()
    with mark('initialization'):
        cfg, model, post = build_runtime(args.config, args.weights, args.device)
    print(READY_MARKER, flush=True)
    request = read_worker_request()
    with mark('inference'):
        count = run_inference(cfg, model, post, args.device, args.batch_size,
                              request['image_dir'], request['meta_file'], request['output_file'])
    print(f'wrote {count} detections (NVTX markers only)', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
