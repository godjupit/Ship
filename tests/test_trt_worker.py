"""Standalone preprocessing, serialization and bounded tail-batch integration."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from supership.trt_worker import load_image, serialize

ROOT = Path(__file__).resolve().parents[1]


class StandaloneCPUChecks(unittest.TestCase):
    def test_reduced_rgb_pixels_match_existing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            pixels = np.random.default_rng(42).integers(0, 256, (35, 47, 3), dtype=np.uint8)
            path = Path(directory) / 'odd.jpg'
            Image.fromarray(pixels).save(path)
            item = dict(file_name=path.name, width=47, height=35)
            for reduction in (1, 2, 4, 8):
                for side in (23, 54):
                    destination = np.empty((side, side, 3), np.uint8)
                    load_image(item, directory, destination, reduction)
                    flag = {1: 1, 2: 17, 4: 33, 8: 65}[reduction]
                    expected = cv2.cvtColor(cv2.resize(cv2.imread(str(path), flag), (side, side)), cv2.COLOR_BGR2RGB)
                    np.testing.assert_array_equal(destination, expected)
                direct = np.empty(((35 + reduction - 1) // reduction,
                                   (47 + reduction - 1) // reduction, 3), np.uint8)
                load_image(item, directory, direct, reduction)
                np.testing.assert_array_equal(direct, cv2.cvtColor(cv2.imread(str(path), flag), cv2.COLOR_BGR2RGB))
            with self.assertRaisesRegex(ValueError, 'metadata dimensions'):
                load_image(dict(item, width=100), directory, destination, 2)

    def test_serialization_order_labels_scale_threshold_and_finite_values(self):
        items = [dict(id=19, width=200, height=100), dict(id=3, width=100, height=200)]
        probabilities = np.array([[[.01, .9, .02]], [[.8, .01, .02]]], np.float32)
        boxes = np.array([[[.5, .5, .2, .4]], [[.5, .5, .4, .2]]], np.float32)
        output = dict(pred_logits=probabilities, pred_boxes=boxes)
        count, body = serialize(items, output, top_queries=1)
        actual = json.loads(b'[' + body + b']')
        self.assertEqual(count, 2)
        self.assertEqual([x['image_id'] for x in actual], [19, 3])
        self.assertEqual([x['category_id'] for x in actual], [1, 0])
        self.assertEqual([x['bbox'] for x in actual], [[80., 30., 40., 40.], [30., 80., 40., 40.]])
        self.assertEqual(serialize(items, output, 1, .95), (0, b''))
        probabilities[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            serialize(items, output)


def available():
    return (shutil.which('nvidia-smi') is not None
            and os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1')
            and (ROOT / 'weights/supership-512-q100.engine').is_file()
            and (ROOT / 'weights/trt_preprocess.ptx').is_file())


@unittest.skipUnless(available(), 'requires CUDA and locally built standalone artifacts')
class StandaloneGPUChecks(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'weights/trt_resize.ptx').exists(), 'requires resize PTX')
    def test_packed_resize_pixels_boundaries_identity_and_zero_tail(self):
        import ctypes
        from cuda.bindings import driver as cu
        from supership.trt_worker import check

        check(cu.cuInit(0))
        context = check(cu.cuDevicePrimaryCtxRetain(0))
        check(cu.cuCtxSetCurrent(context))
        side, batch = 544, 5
        rng = np.random.default_rng(42)
        shapes = [(512, 512), (35, 47), (544, 544), (1, 1)]
        images = [rng.integers(0, 256, (*shape, 3), np.uint8) for shape in shapes]
        packed = np.zeros(batch * 16 + sum(array.nbytes for array in images), np.uint8)
        header = packed[:batch * 16].view(np.int32).reshape(batch, 4)
        offset = batch * 16
        for row, array in enumerate(images):
            header[row, :3] = array.shape[1], array.shape[0], offset
            packed[offset:offset + array.nbytes] = array.reshape(-1)
            offset += array.nbytes
        actual = np.empty((batch, 3, side, side), np.float32)
        source = check(cu.cuMemAlloc(packed.nbytes))
        destination = check(cu.cuMemAlloc(actual.nbytes))
        module = check(cu.cuModuleLoadData((ROOT / 'weights/trt_resize.ptx').read_bytes()))
        try:
            kernel = check(cu.cuModuleGetFunction(module, b'normalize'))
            check(cu.cuMemcpyHtoD(source, packed.ctypes.data, packed.nbytes))
            arguments = [ctypes.c_uint64(int(source)), ctypes.c_uint64(int(destination)),
                         ctypes.c_int(side), ctypes.c_int(batch)]
            parameters = np.array([ctypes.addressof(value) for value in arguments], np.uint64)
            count = actual.size
            check(cu.cuLaunchKernel(kernel, (count + 255) // 256, 1, 1, 256, 1, 1,
                                   0, 0, parameters.ctypes.data, 0))
            check(cu.cuMemcpyDtoH(actual.ctypes.data, destination, actual.nbytes))
            for row, source_image in enumerate(images):
                pixels = np.rint(actual[row].transpose(1, 2, 0) * 255).astype(np.int16)
                expected = cv2.resize(source_image, (side, side), interpolation=cv2.INTER_LINEAR)
                self.assertLessEqual(np.abs(pixels - expected.astype(np.int16)).max(), 1)
                if source_image.shape[:2] in ((side, side), (1, 1)):
                    np.testing.assert_array_equal(pixels, expected)
            np.testing.assert_array_equal(actual[-1], 0)
        finally:
            check(cu.cuModuleUnload(module))
            check(cu.cuMemFree(source))
            check(cu.cuMemFree(destination))
            check(cu.cuDevicePrimaryCtxRelease(0))

    def test_graph_warmup_preserves_recycled_slots_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rng = np.random.default_rng(73)
            items = []
            # More than two complete rings plus a tail catches early slot reuse.
            for i in range(41):
                path = directory / f'{i}.jpg'
                Image.fromarray(rng.integers(0, 256, (63, 79, 3), np.uint8)).save(path)
                items.append(dict(id=100-i, file_name=path.name, width=79, height=63))
            metadata = directory / 'meta.json'
            metadata.write_text(json.dumps(dict(images=items)))
            outputs = []
            for warmup in ('0', '1'):
                output = directory / f'{warmup}.json'
                env = dict(os.environ, BACKEND='tensorrt', TRT_STANDALONE='1',
                           TRT_ENGINE='weights/supership-512-q100.engine', INPUT_SIZE='512',
                           BATCH_SIZE='8', NUM_WORKERS='2', PIPELINE_DEPTH='2',
                           SCORE_THRESHOLD='0', POST_TOP_QUERIES='100',
                           TRT_GRAPH_WARMUP=warmup,
                           PYTHON_BIN=str(ROOT / '.venv/bin/python'))
                result = subprocess.run(
                    ['bash', 'run.sh', '--worker'], cwd=ROOT, env=env,
                    input=json.dumps(dict(image_dir=str(directory), meta_file=str(metadata),
                                          output_file=str(output))) + '\n',
                    capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs.append(output.read_bytes())
            self.assertEqual(outputs[0], outputs[1])
            # Threshold zero makes this a numerical check even on synthetic images.
            self.assertEqual({row['image_id'] for row in json.loads(outputs[0])},
                             {item['id'] for item in items})

    def test_ready_precedes_metadata_access_and_tail_matches_existing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image_dir = directory / 'images'
            image_dir.mkdir()
            items = []
            rng = np.random.default_rng(12)
            for i in range(9):
                path = image_dir / f'{i}.jpg'
                width, height = 47 + i * 2, 35 + i * 2
                Image.fromarray(rng.integers(0, 256, (height, width, 3), np.uint8)).save(path)
                items.append(dict(id=100 - i, file_name=path.name, width=width, height=height))
            env = dict(os.environ, BACKEND='tensorrt', TRT_ENGINE='weights/supership-512-q100.engine',
                       INPUT_SIZE='512', BATCH_SIZE='8', NUM_WORKERS='2', PIPELINE_DEPTH='2',
                       GPU_RESIZE='0',
                       JPEG_REDUCTION='2', SCORE_THRESHOLD='0.05', POST_TOP_QUERIES='100',
                       PYTHON_BIN=str(ROOT / '.venv/bin/python'))
            outputs = []
            for standalone in ('0', '1'):
                metadata = directory / f'meta-{standalone}.json'
                output = directory / f'output-{standalone}.json'
                process = subprocess.Popen(['bash', 'run.sh', '--worker'], cwd=ROOT,
                                           env=dict(env, TRT_STANDALONE=standalone),
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True)
                try:
                    self.assertEqual(process.stdout.readline().strip(), 'THU_RS_READY')
                    # The request and metadata do not exist until after READY.
                    metadata.write_text(json.dumps(dict(images=items)))
                    request = dict(image_dir=str(image_dir), meta_file=str(metadata), output_file=str(output))
                    stdout, stderr = process.communicate(json.dumps(request) + '\n', timeout=120)
                    self.assertEqual(process.returncode, 0, stderr)
                    outputs.append(json.loads(output.read_text()))
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            self.assertEqual(outputs[0], outputs[1])

    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            metadata = directory / 'meta.json'
            output = directory / 'output.json'
            metadata.write_text('{"images":[]}')
            env = dict(os.environ, BACKEND='tensorrt', TRT_STANDALONE='1',
                       TRT_ENGINE='weights/supership-512-q100.engine', INPUT_SIZE='512',
                       PYTHON_BIN=str(ROOT / '.venv/bin/python'))
            result = subprocess.run(['bash', 'run.sh', '--worker'], cwd=ROOT, env=env,
                                    input=json.dumps(dict(image_dir=str(directory), meta_file=str(metadata),
                                                          output_file=str(output))) + '\n',
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text()), [])
