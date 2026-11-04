"""Bounded threaded image pipeline; all dataset access starts after READY."""

import concurrent.futures
from collections import deque
import heapq
import json
import os
from pathlib import Path
import sys
import threading

sys.path.insert(0, os.getcwd())

import numpy as np
import orjson
from PIL import Image
import torch

from supership.generate_submission import build_runtime, model_forward, read_worker_request


def bulk_results(items, chunks, threshold):
    """Round and format all snapshots in input/query order, including short tails."""
    if not chunks:
        return []
    output = np.concatenate(chunks, axis=0)
    if output.shape[0] != len(items) or not np.isfinite(output).all():
        raise ValueError("incomplete or nonfinite bulk detection output")
    ids = np.repeat(np.asarray([item["id"] for item in items]), output.shape[1])
    flat = output.reshape(-1, 6)
    mask = flat[:, 4] >= threshold
    selected = flat[mask].astype(np.float64)
    ids = ids[mask]
    selected[:, 2:4] -= selected[:, :2]
    boxes = np.round(selected[:, :4], 2)
    scores = np.round(selected[:, 4], 5)
    return [dict(image_id=int(image_id), category_id=int(label), bbox=box, score=score)
            for image_id, label, box, score in
            zip(ids, selected[:, 5], boxes.tolist(), scores.tolist())
            if box[2] > 0 and box[3] > 0]


def serialized_batch(items, output, threshold):
    results = bulk_results(items, [output], threshold)
    return len(results), orjson.dumps(results)[1:-1]


def ordered_items(items, image_dir, batch, mode):
    """Schedule from file sizes obtained only after receiving the timed request."""
    if mode == "metadata" or not items:
        return items
    if mode not in ("small-first", "large-first", "balanced"):
        raise ValueError("unsupported INPUT_ORDER")
    weighted = [(Path(image_dir, item["file_name"]).stat().st_size, index, item)
                for index, item in enumerate(items)]
    if mode != "balanced":
        return [entry[2] for entry in sorted(weighted, key=lambda x: x[0],
                                             reverse=mode == "large-first")]
    count = (len(items) + batch - 1) // batch
    bins = [[] for _ in range(count)]
    heap = [(0, index) for index in range(count)]
    heapq.heapify(heap)
    for weight, _, item in sorted(weighted, key=lambda x: (-x[0], x[1])):
        cost, index = heapq.heappop(heap)
        bins[index].append(item)
        capacity = min(batch, len(items) - index * batch)
        if len(bins[index]) < capacity:
            heapq.heappush(heap, (cost + weight, index))
    return [item for group in bins for item in group]


class Runtime:
    def __init__(self):
        self.side = int(os.environ.get("INPUT_SIZE", "640"))
        self.upload_side = int(os.environ.get("UPLOAD_SIZE", str(self.side)))
        self.batch = int(os.environ.get("BATCH_SIZE", "8"))
        self.workers = int(os.environ.get("NUM_WORKERS", "8"))
        self.prefetch = int(os.environ.get("PIPELINE_DEPTH", "3"))
        self.device = os.environ.get("DEVICE", "cuda:0")
        self.precision = os.environ.get("PRECISION", "fp32")
        self.backend = os.environ.get("BACKEND", "cuda_graph")
        self.threshold = float(os.environ.get("SCORE_THRESHOLD", "0"))
        self.decoder = os.environ.get("IMAGE_DECODER", "pil")
        self.reduction = int(os.environ.get("JPEG_REDUCTION", "1"))
        self.copy_mode = os.environ.get("OPENCV_COPY_MODE", "strided")
        self.async_output = os.environ.get("ASYNC_OUTPUT", "0") == "1"
        self.eval_layers = int(os.environ.get("DECODER_LAYERS", "0"))
        self.antialias = os.environ.get("RESIZE_ANTIALIAS", "1") == "1"
        self.vectorized = os.environ.get("VECTORIZED_RESULTS", "0") == "1"
        self.early_recycle = os.environ.get("EARLY_INPUT_RECYCLE", "0") == "1"
        self.bulk_results = os.environ.get("BULK_RESULTS", "0") == "1"
        self.threaded_json = os.environ.get("THREADED_JSON", "0") == "1"
        if self.threaded_json and not self.bulk_results:
            raise ValueError("threaded JSON requires bulk results")
        self.pinned_sizes = os.environ.get("PINNED_SIZES", "0") == "1"
        self.reuse_events = os.environ.get("REUSE_EVENTS", "0") == "1"
        self.fused_normalize = os.environ.get("FUSED_NORMALIZE", "0") == "1"
        self.fused_postprocess = os.environ.get("FUSED_POSTPROCESS", "0") == "1"
        self.gpu_bgr = os.environ.get("GPU_BGR", "0") == "1"
        if self.gpu_bgr and (self.decoder != "opencv" or not self.fused_normalize):
            raise ValueError("GPU BGR conversion requires OpenCV decoding and fused normalization")
        if self.fused_normalize and (self.precision != "fp16" or os.environ.get("CHANNELS_LAST", "1") != "1"):
            raise ValueError("fused normalization requires channels-last FP16 inference")
        if self.upload_side < 320 or self.upload_side > self.side or self.upload_side % 32:
            raise ValueError("UPLOAD_SIZE must be a multiple of 32 between 320 and INPUT_SIZE")
        if self.upload_side != self.side and (self.decoder == "cuda" or self.fused_normalize):
            raise ValueError("reduced upload requires CPU decoding and native normalization")
        if self.early_recycle and self.decoder == "cuda":
            raise ValueError("early input recycling requires CPU image decoding")
        if self.bulk_results and not self.vectorized:
            raise ValueError("bulk results requires VECTORIZED_RESULTS=1")
        if self.side < 320 or self.side > 1280 or self.side % 32:
            raise ValueError("INPUT_SIZE must be a multiple of 32 between 320 and 1280")
        if min(self.batch, self.workers, self.prefetch) < 1:
            raise ValueError("batch, workers and pipeline depth must be positive")
        if self.precision not in ("fp32", "fp16") or self.backend not in ("torch", "cuda_graph"):
            raise ValueError("unsupported precision or backend")
        if not 0 <= self.threshold <= 1:
            raise ValueError("score threshold must be in [0, 1]")
        if self.decoder not in ("pil", "opencv", "cuda"):
            raise ValueError("unsupported image decoder")
        if self.reduction not in (1, 2, 4, 8) or (self.reduction != 1 and self.decoder != "opencv"):
            raise ValueError("JPEG_REDUCTION must be 1, 2, 4 or 8; reduction requires OpenCV")
        if self.copy_mode not in ("strided", "cvtcolor", "direct"):
            raise ValueError("unsupported OPENCV_COPY_MODE")
        if self.eval_layers not in (0, 1, 2, 3):
            raise ValueError("DECODER_LAYERS must be 0 (checkpoint default), 1, 2 or 3")
        torch.backends.cudnn.benchmark = os.environ.get("CUDNN_BENCHMARK", "0") == "1"
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
        if self.decoder == "opencv":
            import cv2
            cv2.setNumThreads(1)
        if self.decoder == "cuda":
            # Initialize nvJPEG using a synthetic JPEG, without opening any dataset.
            import io
            from torchvision.io import decode_jpeg
            from torchvision.transforms.v2.functional import resize
            buffer = io.BytesIO()
            Image.new("RGB", (64, 64), (127, 127, 127)).save(buffer, format="JPEG")
            encoded = torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)
            synthetic = decode_jpeg([encoded], mode="RGB", device=self.device)[0]
            resize(synthetic, [640, 640], antialias=self.antialias)
            torch.cuda.synchronize(self.device)
        _, self.model, self.post = build_runtime(
            os.environ.get("MODEL_CONFIG", "configs/dfine/dfine_hgnetv2_s_supership.yml"),
            os.environ.get("MODEL_WEIGHTS", "weights/baseline.pth"), self.device,
        )
        if self.side != 640:
            # Load all trained weights strictly first, then regenerate only geometry.
            encoder, decoder = self.model.encoder, self.model.decoder
            encoder.eval_spatial_size = decoder.eval_spatial_size = (self.side, self.side)
            for index in encoder.use_encoder_idx:
                stride = encoder.feat_strides[index]
                position = encoder.build_2d_sincos_position_embedding(
                    self.side // stride, self.side // stride,
                    encoder.hidden_dim, encoder.pe_temperature, device=self.device)
                setattr(encoder, f"pos_embed{index}", position)
            decoder.anchors, decoder.valid_mask = decoder._generate_anchors(device=self.device)
        if self.eval_layers:
            # Keep trained head dimensions; change only the inference stopping layer.
            if self.eval_layers > self.model.decoder.num_layers:
                raise ValueError("DECODER_LAYERS exceeds the checkpoint architecture")
            self.model.decoder.eval_idx = self.eval_layers - 1
            self.model.decoder.decoder.eval_idx = self.eval_layers - 1
        queries = int(os.environ.get("NUM_QUERIES", "0"))
        if queries:
            if not 1 <= queries <= self.model.decoder.num_queries or self.model.decoder.learn_query_content:
                raise ValueError("query reduction requires unlearned query content and a valid count")
            self.model.decoder.num_queries = queries
        top_queries = int(os.environ.get("POST_TOP_QUERIES", "0"))
        if top_queries:
            if not 1 <= top_queries <= self.model.decoder.num_queries * self.post.num_classes:
                raise ValueError("invalid output query count")
            self.post.num_top_queries = top_queries
        if self.fused_postprocess and (not self.post.use_focal_loss or self.post.remap_mscoco_category):
            raise ValueError("fused postprocessing requires focal scores without category remapping")
        if os.environ.get("MODEL_DEPLOY", "0") == "1":
            self.model.deploy()
            self.model.eval()
        if os.environ.get("FUSE_CONV_BN", "0") == "1":
            from supership.conv_bn_fusion import fuse_model_conv_bn
            fused = fuse_model_conv_bn(self.model)
            print(f"Conv-BN fused pairs: {len(fused)}", file=sys.stderr, flush=True)
        from supership.kernel_optimizations import apply_kernel_options
        changes = apply_kernel_options(self.model, self.precision)
        if changes:
            print(f"Kernel options: {changes}", file=sys.stderr, flush=True)
        if os.environ.get("CHANNELS_LAST", "1") == "1":
            self.model.to(memory_format=torch.channels_last)
        self.post.deploy()
        self.images = torch.zeros(self.batch, self.upload_side, self.upload_side, 3, dtype=torch.uint8, device=self.device)
        self.sizes = torch.ones(self.batch, 2, dtype=torch.float32, device=self.device)
        self.host = [torch.empty_like(self.images, device="cpu", pin_memory=True)
                     for _ in range(self.prefetch)]
        self.host_arrays = [x.numpy() for x in self.host]
        if self.pinned_sizes:
            self.host_sizes = [torch.empty(self.batch, 2, dtype=torch.float32, pin_memory=True)
                               for _ in range(self.prefetch)]
            self.size_arrays = [x.numpy() for x in self.host_sizes]
        if self.reuse_events:
            self.copy_events = [torch.cuda.Event() for _ in range(self.prefetch)]
            self.output_events = [torch.cuda.Event() for _ in range(self.prefetch)]
        self.output_stream = torch.cuda.Stream(device=self.device) if self.async_output else None
        self.host_outputs = None
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        # Start threads using synthetic work only, without accessing evaluation paths.
        barrier = threading.Barrier(self.workers)
        jobs = [self.pool.submit(barrier.wait) for _ in range(self.workers)]
        for job in jobs:
            job.result()
        self.result_pool = None
        if self.threaded_json:
            self.result_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self.result_pool.submit(lambda: None).result()
        with torch.inference_mode():
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(5):
                    self.forward()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            self.graph = None
            if self.backend == "cuda_graph":
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.captured = self.forward()
                torch.cuda.synchronize(self.device)
            if self.async_output:
                example = self.captured if self.graph is not None else self.forward()
                self.host_outputs = [torch.empty_like(example, device="cpu", pin_memory=True)
                                     for _ in range(self.prefetch)]
                torch.cuda.synchronize(self.device)

    def forward(self):
        if self.fused_normalize:
            from supership.fused_io import normalized_half
            images = normalized_half(self.images, self.gpu_bgr)
        else:
            images = self.images.permute(0, 3, 1, 2).float()
            if self.upload_side != self.side:
                images = torch.nn.functional.interpolate(images, size=(self.side, self.side),
                                                        mode="bilinear", align_corners=False)
                images.round_()
            images.mul_(1 / 255)
        if os.environ.get("CHANNELS_LAST", "1") != "1":
            images = images.contiguous()
        outputs = model_forward(self.model, images, self.precision)
        if self.fused_postprocess:
            from supership.fused_io import packed_detections
            return packed_detections(outputs, self.sizes, self.post.num_top_queries)
        labels, boxes, scores = self.post(outputs, self.sizes)
        return torch.cat((boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)), dim=-1)

    def load_image(self, item, image_dir, destination):
        # image_dir and metadata are obtained only after READY.
        path = Path(image_dir) / item["file_name"]
        if self.decoder == "cuda":
            return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
        if self.decoder == "opencv":
            import cv2
            flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
                    4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[self.reduction]
            image = cv2.imread(str(path), flag)
            if image is None:
                raise ValueError(f"cannot decode image: {path}")
            h, w = image.shape[:2]
            copy_mode = getattr(self, "copy_mode", "strided")
            gpu_bgr = getattr(self, "gpu_bgr", False)
            if copy_mode != "direct" and not gpu_bgr:
                resized = (image if image.shape[:2] == destination.shape[:2] else
                           cv2.resize(image, (destination.shape[1], destination.shape[0]), interpolation=cv2.INTER_LINEAR))
            if gpu_bgr:
                cv2.resize(image, (destination.shape[1], destination.shape[0]), dst=destination, interpolation=cv2.INTER_LINEAR)
            elif copy_mode == "direct":
                # Reuse the pinned destination for both resize and channel swap.
                cv2.resize(image, (destination.shape[1], destination.shape[0]), dst=destination, interpolation=cv2.INTER_LINEAR)
                cv2.cvtColor(destination, cv2.COLOR_BGR2RGB, dst=destination)
            elif copy_mode == "cvtcolor":
                cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=destination)
            else:
                destination[:] = resized[:, :, ::-1]
        else:
            with Image.open(path) as image:
                image = image.convert("RGB")
                w, h = image.size
                destination[:] = np.asarray(image.resize((destination.shape[1], destination.shape[0]), Image.Resampling.BILINEAR))
        reduction = self.reduction if self.decoder == "opencv" else 1
        expected = tuple((item[key] + reduction - 1) // reduction for key in ("width", "height"))
        if (w, h) != expected:
            raise ValueError(f"metadata dimensions disagree with decoded image: {path}")
        return item["width"], item["height"]

    def load_after_copy(self, event, item, image_dir, destination):
        # Reuse pinned input only after its H2D transfer, independently of D2H.
        event.synchronize()
        return self.load_image(item, image_dir, destination)

    def run(self, request):
        metadata = json.loads(Path(request["meta_file"]).read_text())
        items = metadata["images"]
        ids = [item["id"] for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate image ids")
        items = ordered_items(items, request["image_dir"], self.batch,
                              os.environ.get("INPUT_ORDER", "metadata"))
        jobs = {}
        count = (len(items) + self.batch - 1) // self.batch

        def schedule(index, dependency=None):
            slot = index % self.prefetch
            batch = items[index * self.batch:(index + 1) * self.batch]
            loader = self.load_image if dependency is None else self.load_after_copy
            prefix = () if dependency is None else (dependency,)
            jobs[index] = [self.pool.submit(loader, *prefix, item, request["image_dir"],
                                           self.host_arrays[slot][row])
                           for row, item in enumerate(batch)]

        for index in range(min(self.prefetch, count)):
            schedule(index)
        results = []
        snapshots = []
        output_jobs = []
        pending = deque()

        def collect(batch, output):
            if not np.isfinite(output).all():
                raise ValueError("nonfinite detection")
            if self.threaded_json:
                output_jobs.append(self.result_pool.submit(serialized_batch, batch, output.copy(), self.threshold))
                return
            if self.bulk_results:
                snapshots.append(output.copy())
                return
            for item, detections in zip(batch, output):
                selected = detections[detections[:, 4] >= self.threshold].astype(np.float64)
                if self.vectorized:
                    selected[:, 2:4] -= selected[:, :2]
                    boxes = np.round(selected[:, :4], 2)
                    scores = np.round(selected[:, 4], 5)
                    for box, score, label in zip(boxes.tolist(), scores.tolist(), selected[:, 5]):
                        if box[2] > 0 and box[3] > 0:
                            results.append(dict(image_id=int(item["id"]), category_id=int(label),
                                                bbox=box, score=score))
                else:
                    for x1, y1, x2, y2, score, label in selected.tolist():
                        w, h = round(x2 - x1, 2), round(y2 - y1, 2)
                        if w > 0 and h > 0:
                            results.append(dict(image_id=int(item["id"]), category_id=int(label),
                                                bbox=[round(x1, 2), round(y1, 2), w, h],
                                                score=round(score, 5)))
        with torch.inference_mode():
            for index in range(count):
                slot = index % self.prefetch
                batch = items[index * self.batch:(index + 1) * self.batch]
                loaded = [job.result() for job in jobs.pop(index)]
                n = len(batch)
                if self.decoder == "cuda":
                    from torchvision.io import decode_jpeg
                    from torchvision.transforms.v2.functional import resize
                    decoded = decode_jpeg(loaded, mode="RGB", device=self.device)
                    sizes = []
                    groups = {}
                    for row, (item, image) in enumerate(zip(batch, decoded)):
                        h, w = image.shape[-2:]
                        if (w, h) != (item["width"], item["height"]):
                            raise ValueError("metadata dimensions disagree with decoded image")
                        sizes.append((w, h))
                        groups.setdefault((h, w), []).append(row)
                    for indices in groups.values():
                        images = torch.stack([decoded[row] for row in indices])
                        resized = resize(images, [self.side, self.side], antialias=self.antialias)
                        for row, image in zip(indices, resized):
                            self.images[row].copy_(image.permute(1, 2, 0))
                    if n < self.batch:
                        self.images[n:].zero_()
                else:
                    sizes = loaded
                    if n < self.batch:
                        self.host[slot][n:].zero_()
                    self.images.copy_(self.host[slot], non_blocking=True)
                    if self.early_recycle and index + self.prefetch < count:
                        copied = self.copy_events[slot] if self.reuse_events else torch.cuda.Event()
                        copied.record(torch.cuda.current_stream(self.device))
                        schedule(index + self.prefetch, dependency=copied)
                if self.pinned_sizes:
                    # Loaded jobs validate dimensions before these values are copied.
                    # Reuse is fenced by completion of this slot's previous output.
                    self.size_arrays[slot][:n] = sizes
                    self.sizes[:n].copy_(self.host_sizes[slot][:n], non_blocking=True)
                else:
                    self.sizes[:n].copy_(torch.tensor(sizes, dtype=torch.float32), non_blocking=True)
                if n < self.batch:
                    self.sizes[n:].fill_(1)
                if self.graph is not None:
                    self.graph.replay()
                    device_output = self.captured
                else:
                    device_output = self.forward()
                if self.async_output:
                    # Copy off the graph's static output before its next replay.
                    # The next batch can execute while Python formats older results.
                    current = torch.cuda.current_stream(self.device)
                    self.output_stream.wait_stream(current)
                    with torch.cuda.stream(self.output_stream):
                        self.host_outputs[slot].copy_(device_output, non_blocking=True)
                        ready = self.output_events[slot] if self.reuse_events else torch.cuda.Event()
                        ready.record(self.output_stream)
                    current.wait_event(ready)
                    pending.append((index, batch, ready))
                    if len(pending) >= min(2, self.prefetch):
                        previous, previous_batch, event = pending.popleft()
                        event.synchronize()
                        previous_slot = previous % self.prefetch
                        # This completion also fences the pinned input's H2D copy.
                        if not self.early_recycle and previous + self.prefetch < count:
                            schedule(previous + self.prefetch)
                        collect(previous_batch, self.host_outputs[previous_slot][:len(previous_batch)].numpy())
                else:
                    output = device_output[:n].cpu().numpy()
                    if not self.early_recycle and index + self.prefetch < count:
                        schedule(index + self.prefetch)
                    collect(batch, output)
            for previous, previous_batch, event in pending:
                event.synchronize()
                collect(previous_batch, self.host_outputs[previous % self.prefetch][:len(previous_batch)].numpy())
        self.pool.shutdown(wait=True)
        torch.cuda.synchronize(self.device)
        if self.threaded_json:
            try:
                chunks = [job.result() for job in output_jobs]
            finally:
                self.result_pool.shutdown(wait=True)
            detections = sum(n for n, _ in chunks)
            payload = b"[" + b",".join(part for _, part in chunks if part) + b"]"
        else:
            if self.bulk_results:
                results = bulk_results(items, snapshots, self.threshold)
            # Nonfinite model outputs were rejected before orjson serialization.
            payload = orjson.dumps(results)
            detections = len(results)
        path = Path(request["output_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        return detections, len(items)


def main():
    runtime = Runtime()
    print("THU_RS_READY", flush=True)
    detections, images = runtime.run(read_worker_request())
    print(f"processed {images} images; wrote {detections} detections", file=sys.stderr, flush=True)
    # All threads joined, kernels synchronized, and output closed after fsync.
    # Avoid spending timed inference on Python/CUDA allocator destruction.
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("FAST_EXIT", "1") == "1":
        os._exit(0)


if __name__ == "__main__":
    main()
