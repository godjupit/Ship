"""RGB codec decoding must preserve the existing preprocessing pixels."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from supership.fast_worker import Runtime


class RGBDecodeTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(cv2, 'IMREAD_COLOR_RGB'), 'OpenCV RGB decoding unavailable')
    def test_codec_pixels_and_coordinate_scale(self):
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as directory:
            for mode, suffix in [('RGB', 'jpg'), ('L', 'jpg'), ('CMYK', 'jpg'),
                                 ('RGB', 'png'), ('RGBA', 'png')]:
                height, width = (35, 47) if suffix == "jpg" else (40, 48)
                image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).convert(mode)
                path = Path(directory) / f'{mode}.{suffix}'
                image.save(path)
                item = dict(file_name=path.name, width=width, height=height)
                for reduction in (1, 2, 4, 8):
                    with self.subTest(mode=mode, suffix=suffix, reduction=reduction):
                        runtime = Runtime.__new__(Runtime)
                        runtime.decoder = 'opencv'
                        runtime.reduction = reduction
                        runtime.gpu_bgr = False
                        expected = np.empty((54, 54, 3), dtype=np.uint8)
                        actual = np.empty_like(expected)
                        runtime.copy_mode = 'cvtcolor'
                        runtime.load_image(item, directory, expected)
                        runtime.copy_mode = 'rgb'
                        self.assertEqual(runtime.load_image(item, directory, actual), (width, height))
                        np.testing.assert_array_equal(actual, expected)
                        with self.assertRaisesRegex(ValueError, 'metadata dimensions'):
                            runtime.load_image(dict(item, width=100), directory, actual)
                        runtime.gpu_bgr = True
                        runtime.load_image(item, directory, actual)
                        np.testing.assert_array_equal(actual, expected[:, :, ::-1])
