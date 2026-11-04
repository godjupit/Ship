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
