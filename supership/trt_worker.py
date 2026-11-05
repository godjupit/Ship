"""Bounded TensorRT/CUDA Driver pipeline without importing PyTorch.

Evaluation paths are read only after READY. Engine and PTX artifacts are built
separately, so deployment requires neither a compiler nor network access.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supership.submission import load_json


def check(result):
    """Check a CUDA Driver result before returning its output values."""
    if int(result[0]):
        raise RuntimeError(f'CUDA Driver call failed: {result}')
    return result[1] if len(result) == 2 else result[1:]


def load_image(item, directory, destination, reduction):
    """Decode into pinned RGB; resize on CPU only for the bounded-buffer fallback."""
    path = Path(directory) / item['file_name']
    expected = tuple((item[k] + reduction - 1) // reduction for k in ('height', 'width'))
    flag = cv2.IMREAD_COLOR_RGB | {1: 0, 2: 16, 4: 32, 8: 64}[reduction]
    if destination.shape[:2] == expected:
        try:
            image = cv2.imread(str(path), destination, flag)
        except cv2.error:
            # Non-JPEG codecs can allocate full size and round reductions down.
            image = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)
            if image is not None and image.shape[:2] == (item['height'], item['width']):
                image = cv2.resize(image, (expected[1], expected[0]))
    else:
        image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f'cannot decode image: {path}')
    if image.shape[:2] != expected:
        full = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)
        if full is not None and full.shape[:2] == (item['height'], item['width']):
            image = cv2.resize(full, (expected[1], expected[0]))
    if image.shape[:2] != expected:
        raise ValueError(f'metadata dimensions disagree with decoded image: {path}')
    if image.shape == destination.shape:
        if image.ctypes.data != destination.ctypes.data:
            np.copyto(destination, image)
    else:
        cv2.resize(image, (destination.shape[1], destination.shape[0]),
                   dst=destination, interpolation=cv2.INTER_LINEAR)


def serialize(items, output, top_queries=100, threshold=0.05):
    """Select GPU-computed probabilities and preserve the existing JSON rounding."""
    probabilities, boxes = output['pred_logits'], output['pred_boxes']
    if not np.isfinite(probabilities).all() or not np.isfinite(boxes).all():
        raise ValueError('nonfinite detection')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('invalid detection probability')
    flat = probabilities.reshape(len(items), -1)
    index = np.argsort(-flat, axis=1, kind='stable')[:, :top_queries]
    scores = np.take_along_axis(flat, index, axis=1)
    labels, queries = index % probabilities.shape[-1], index // probabilities.shape[-1]
    corners = np.concatenate((boxes[:, :, :2] - boxes[:, :, 2:] / 2,
                              boxes[:, :, :2] + boxes[:, :, 2:] / 2), axis=2)
    sizes = np.array([[item['width'], item['height']] * 2 for item in items], dtype=np.float32)
    corners *= sizes[:, None, :]
    corners = np.take_along_axis(corners, queries[:, :, None], axis=1)
    packed = np.concatenate((corners, scores[:, :, None],
                             labels[:, :, None].astype(np.float32)), axis=2)
    selected = packed.reshape(-1, 6)
    ids = np.repeat([item['id'] for item in items], index.shape[1])
    mask = selected[:, 4] >= threshold
    selected, ids = selected[mask].astype(np.float64), ids[mask]
    selected[:, 2:4] -= selected[:, :2]
    coordinates, scores = np.round(selected[:, :4], 2), np.round(selected[:, 4], 5)
    results = [dict(image_id=int(image_id), category_id=int(label), bbox=box, score=score)
               for image_id, label, box, score in
               zip(ids, selected[:, 5], coordinates.tolist(), scores.tolist())
               if box[2] > 0 and box[3] > 0]
    return len(results), orjson.dumps(results)[1:-1]


class Runtime:
    def __init__(self):
        import tensorrt as trt
        from cuda.bindings import driver as cu

        self.cu = cu
        cv2.setNumThreads(1)
        device = os.environ.get('DEVICE', 'cuda:0')
        if device != 'cuda:0':
            raise ValueError('standalone TensorRT currently requires DEVICE=cuda:0')
        check(cu.cuInit(0))
        self.context_handle = check(cu.cuDevicePrimaryCtxRetain(0))
        check(cu.cuCtxSetCurrent(self.context_handle))
        self.stream = check(cu.cuStreamCreate(1))
        path = Path(os.environ.get('TRT_ENGINE', 'weights/supership-544.engine'))
        data = path.read_bytes()
        metadata = json.loads(Path(str(path) + '.json').read_text())
        weights = Path(os.environ.get('MODEL_WEIGHTS', 'weights/baseline.pth'))
        gpu_name = check(cu.cuDeviceGetName(256, 0)).split(b'\0')[0].decode()
        if metadata.get('engine_sha256') != hashlib.sha256(data).hexdigest():
            raise ValueError('TensorRT engine checksum mismatch')
        if metadata.get('weights_sha256') != hashlib.sha256(weights.read_bytes()).hexdigest():
            raise ValueError('TensorRT engine was built from different weights')
        if metadata.get('tensorrt') != trt.__version__ or metadata.get('gpu') != gpu_name:
            raise ValueError('TensorRT version or GPU changed; rebuild the engine')
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError('TensorRT engine deserialization failed')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT context creation failed')
        self.buffers, self.shapes = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if not shape or any(d <= 0 for d in shape) or self.engine.get_tensor_dtype(name) != trt.float32:
                raise ValueError(f'expected static FP32 engine I/O, got {name}: {shape}')
            self.shapes[name] = shape
            pointer = check(cu.cuMemAlloc(int(np.prod(shape)) * 4))
            self.buffers[name] = pointer
            if not self.context.set_tensor_address(name, int(pointer)):
                raise RuntimeError(f'cannot bind engine tensor: {name}')
        if set(self.shapes) != {'images', 'pred_logits', 'pred_boxes'}:
            raise ValueError('unexpected TensorRT I/O tensors')
        self.batch = int(os.environ.get('BATCH_SIZE', '8'))
        self.side = int(os.environ.get('INPUT_SIZE', '544'))
        if self.shapes['images'] != (self.batch, 3, self.side, self.side):
            raise ValueError('TensorRT input does not match BATCH_SIZE/INPUT_SIZE')
        output_shape = self.shapes['pred_logits']
        if (len(output_shape) != 3 or output_shape[0] != self.batch or output_shape[2] != 3
                or self.shapes['pred_boxes'] != (*output_shape[:2], 4)):
            raise ValueError('expected three-class logits and matching box predictions')
        self.depth = int(os.environ.get('PIPELINE_DEPTH', '3'))
        self.reduction = int(os.environ.get('JPEG_REDUCTION', '2'))
        self.workers = int(os.environ.get('NUM_WORKERS', '10'))
        self.top_queries = int(os.environ.get('POST_TOP_QUERIES', '100'))
        self.threshold = float(os.environ.get('SCORE_THRESHOLD', '.05'))
        if self.depth < 2 or self.workers < 1 or self.reduction not in (1, 2, 4, 8):
            raise ValueError('invalid pipeline depth, worker count or JPEG reduction')
        if not 1 <= self.top_queries <= int(np.prod(output_shape[1:])) or not 0 <= self.threshold <= 1:
            raise ValueError('invalid top-query count or score threshold')
        if int(os.environ.get('DECODER_LAYERS', '0')) or int(os.environ.get('NUM_QUERIES', '0')):
            raise ValueError('TensorRT query/layer counts are fixed at build time')
        self.max_pixels = max(self.side * self.side, int(os.environ.get('MAX_DECODE_PIXELS', str(self.side * self.side))))
        self.gpu_resize = os.environ.get('GPU_RESIZE', '1') == '1'
        if self.batch * (16 + self.max_pixels * 3) >= 2**31:
            raise ValueError('packed input offsets must fit signed 32-bit integers')
        # Each bounded slot starts with one [width, height, byte offset, format]
        # descriptor per image, followed by tightly packed RGB pixels.
        self.host = [self.host_array((self.batch * (16 + self.max_pixels * 3),), np.uint8)
                     for _ in range(self.depth)]
        self.used = [0] * self.depth
        self.outputs = [{name: self.host_array(self.shapes[name], np.float32)
                         for name in ('pred_logits', 'pred_boxes')} for _ in range(self.depth)]
        self.raw = check(cu.cuMemAlloc(self.host[0].nbytes))
        check(cu.cuMemsetD8(self.raw, 0, self.host[0].nbytes))
        decoder_mode = os.environ.get('STANDALONE_DECODER', 'opencv')
        if decoder_mode not in ('opencv', 'rgb', 'yuv'):
            raise ValueError('invalid standalone decoder')
        self.decoder = None
        if decoder_mode != 'opencv':
            from supership.jpeg_decoder import JPEGDecoder
            self.decoder = JPEGDecoder(load_image, decoder_mode)
        kernel_name = 'trt_yuv' if decoder_mode == 'yuv' else 'trt_resize'
        ptx_path = Path(os.environ.get('TRT_KERNELS', f'weights/{kernel_name}.ptx'))
        ptx = ptx_path.read_bytes()
        kernel_metadata = json.loads(Path(str(ptx_path) + '.json').read_text())
        source = Path(__file__).with_name(kernel_name + '.cu').read_bytes()
        if (kernel_metadata.get('ptx_sha256') != hashlib.sha256(ptx).hexdigest()
                or kernel_metadata.get('source_sha256') != hashlib.sha256(source).hexdigest()):
            raise ValueError('CUDA kernel checksum mismatch; run scripts/build_trt_kernels.py')
        self.module = check(cu.cuModuleLoadData(ptx))
        self.kernel = check(cu.cuModuleGetFunction(self.module, b'normalize'))
        self.arguments = [ctypes.c_uint64(int(self.raw)), ctypes.c_uint64(int(self.buffers['images'])),
                          ctypes.c_int(self.side), ctypes.c_int(self.batch)]
        self.parameters = np.array([ctypes.addressof(a) for a in self.arguments], dtype=np.uint64)
        self.sigmoid = check(cu.cuModuleGetFunction(self.module, b'sigmoid'))
        self.score_count = int(np.prod(output_shape))
        self.score_arguments = [ctypes.c_uint64(int(self.buffers['pred_logits'])), ctypes.c_int(self.score_count)]
        self.score_parameters = np.array([ctypes.addressof(a) for a in self.score_arguments], dtype=np.uint64)
        # Warm up only synthetic zero inputs; no evaluation paths are known yet.
        for _ in range(5):
            self.forward()
        check(cu.cuStreamSynchronize(self.stream))
        check(cu.cuStreamBeginCapture(self.stream, cu.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL))
        self.forward()
        self.graph = check(cu.cuStreamEndCapture(self.stream))
        self.graph_exec = check(cu.cuGraphInstantiate(self.graph, 0))
        if os.environ.get('TRT_GRAPH_WARMUP', '1') == '1':
            # First graph launch uploads executable state. Use synthetic inputs
            # before READY so this work does not delay the measured request.
            check(cu.cuGraphLaunch(self.graph_exec, self.stream))
            check(cu.cuStreamSynchronize(self.stream))
        self.events = [check(cu.cuEventCreate(2)) for _ in range(self.depth)]
        self.copied = [check(cu.cuEventCreate(2)) for _ in range(self.depth)]
        self.pool = ThreadPoolExecutor(max_workers=self.workers)
        barrier = threading.Barrier(self.workers)
        jobs = [self.pool.submit(barrier.wait) for _ in range(self.workers)]
        for job in jobs:
            job.result()
        self.result_pool = ThreadPoolExecutor(max_workers=1)
        self.result_pool.submit(lambda: None).result()

    def host_array(self, shape, dtype):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        pointer = int(check(self.cu.cuMemHostAlloc(size, 0)))
        # The single-request process owns these pinned allocations until exit.
        storage = (ctypes.c_ubyte * size).from_address(pointer)
        return np.ctypeslib.as_array(storage).view(dtype).reshape(shape)

    def forward(self):
        cu = self.cu
        total = self.batch * self.side * self.side * 3
        check(cu.cuLaunchKernel(self.kernel, (total + 255) // 256, 1, 1, 256, 1, 1,
                               0, self.stream, self.parameters.ctypes.data, 0))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError('TensorRT execution failed')
        check(cu.cuLaunchKernel(self.sigmoid, (self.score_count + 255) // 256, 1, 1,
                               256, 1, 1, 0, self.stream, self.score_parameters.ctypes.data, 0))

    def run(self, request):
        items = load_json(request['meta_file'])['images']
        if not isinstance(items, list):
            raise ValueError('metadata must contain an images array')
        ids = set()
        for item in items:
            if (type(item.get('id')) is not int or item['id'] in ids
                    or not isinstance(item.get('file_name'), str)
                    or any(type(item.get(key)) is not int or item[key] <= 0 for key in ('width', 'height'))):
                raise ValueError('invalid or duplicate image metadata')
            ids.add(item['id'])
        count = (len(items) + self.batch - 1) // self.batch
        jobs, pending, results = {}, deque(), []
        cu = self.cu

        def schedule(index):
            slot = index % self.depth
            batch = items[index * self.batch:(index + 1) * self.batch]
            # Unused tail descriptors remain zero; the GPU writes zero inputs.
            header = self.host[slot][:self.batch * 16].view(np.int32).reshape(self.batch, 4)
            header.fill(0)
            offset = self.batch * 16
            jobs[index] = []
            for row, item in enumerate(batch):
                h, w = ((item[key] + self.reduction - 1) // self.reduction for key in ('height', 'width'))
                if not self.gpu_resize or h * w > self.max_pixels:
                    # Large/unknown image shapes keep the existing CPU resize
                    # path instead of growing pinned memory with the dataset.
                    h = w = self.side
                header[row, :3] = w, h, offset
                destination = self.host[slot][offset:offset + h * w * 3].reshape(h, w, 3)
                if self.decoder is None:
                    job = self.pool.submit(load_image, item, request['image_dir'], destination, self.reduction)
                else:
                    job = self.pool.submit(self.decoder, item, request['image_dir'], destination, self.reduction, header[row])
                jobs[index].append(job)
                offset += h * w * 3
            self.used[slot] = offset

        def collect(index, batch):
            slot = index % self.depth
            # Fence D2H before copying data or reusing an output slot.
            check(cu.cuEventSynchronize(self.events[slot]))
            snapshot = {name: array[:len(batch)].copy() for name, array in self.outputs[slot].items()}
            results.append(self.result_pool.submit(serialize, batch, snapshot, self.top_queries, self.threshold))

        try:
            for i in range(min(self.depth, count)):
                schedule(i)
            for i in range(count):
                slot = i % self.depth
                batch = items[i * self.batch:(i + 1) * self.batch]
                for job in jobs.pop(i):
                    job.result()
                check(cu.cuMemcpyHtoDAsync(self.raw, self.host[slot].ctypes.data,
                                          self.used[slot], self.stream))
                check(cu.cuEventRecord(self.copied[slot], self.stream))
                check(cu.cuGraphLaunch(self.graph_exec, self.stream))
                for name, array in self.outputs[slot].items():
                    check(cu.cuMemcpyDtoHAsync(array.ctypes.data, self.buffers[name], array.nbytes, self.stream))
                check(cu.cuEventRecord(self.events[slot], self.stream))
                pending.append((i, batch))
                if i + self.depth < count:
                    # CPU decoding may overwrite a pinned slot only after H2D.
                    check(cu.cuEventSynchronize(self.copied[slot]))
                    schedule(i + self.depth)
                if len(pending) >= 2:
                    collect(*pending.popleft())
            for arguments in pending:
                collect(*arguments)
        finally:
            self.pool.shutdown(wait=True)
            if self.decoder is not None:
                self.decoder.close()
            check(cu.cuStreamSynchronize(self.stream))
            self.result_pool.shutdown(wait=True)
        chunks = [future.result() for future in results]
        payload = b'[' + b','.join(part for _, part in chunks if part) + b']'
        path = Path(request['output_file'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        return sum(n for n, _ in chunks), len(items)


def main():
    runtime = Runtime()
    print('THU_RS_READY', flush=True)
    request = json.loads(sys.stdin.readline())
    if not isinstance(request, dict) or not {'image_dir', 'meta_file', 'output_file'} <= request.keys():
        raise ValueError('worker request requires image_dir, meta_file and output_file')
    detections, images = runtime.run(request)
    print(f'processed {images} images; wrote {detections} detections', file=sys.stderr, flush=True)
    # All GPU work and CPU futures completed; the fsynced output file is closed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
