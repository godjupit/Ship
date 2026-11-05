"""Direct JPEG destinations, fallback formats, and mixed RGB/YUV CUDA inputs."""
import ctypes
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
from PIL import Image

from supership.jpeg_decoder import JPEGDecoder, has_app1
from supership.trt_worker import load_image, check

ROOT = Path(__file__).resolve().parents[1]


class JPEGDecoderChecks(unittest.TestCase):
    def test_header_parser_stops_before_entropy_data(self):
        app0 = b'\xff\xd8\xff\xe0\x00\x04AB'
        self.assertFalse(has_app1(app0 + b'\xff\xdaExif\x00\x00\xff\xe1'))
        self.assertTrue(has_app1(app0 + b'\xff\xff\xe1\x00\x08Exif\x00\x00'))
        self.assertTrue(has_app1(b'\xff\xd8\xff\xe0\x00\x01'))

    def test_direct_rgb_reductions_and_pinned_destination_bounds(self):
        decoder = JPEGDecoder(load_image, 'rgb')
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'odd.jpg'
                Image.fromarray(np.random.default_rng(4).integers(0, 256, (35, 47, 3), np.uint8)).save(path)
                item = dict(file_name=path.name, height=35, width=47)
                for reduction in (1, 2, 4, 8):
                    h, w = (35 + reduction - 1) // reduction, (47 + reduction - 1) // reduction
                    storage = np.full(h * w * 3 + 32, 177, np.uint8)
                    destination = storage[:-32].reshape(h, w, 3)
                    header = np.zeros(4, np.int32)
                    decoder(item, directory, destination, reduction, header)
                    expected = np.empty_like(destination)
                    load_image(item, directory, expected, reduction)
                    np.testing.assert_array_equal(destination, expected)
                    self.assertTrue((storage[-32:] == 177).all())
                    self.assertEqual(header[3], 0)
                with self.assertRaisesRegex(ValueError, 'metadata dimensions'):
                    decoder(dict(item, width=48), directory, np.empty((18, 24, 3), np.uint8), 2, header)
        finally:
            decoder.close()

    def test_yuv_and_fallback_formats(self):
        fallback = Mock(wraps=load_image)
        decoder = JPEGDecoder(fallback, 'yuv')
        try:
            with tempfile.TemporaryDirectory() as directory:
                pixels = np.random.default_rng(8).integers(0, 256, (35, 47, 3), np.uint8)
                for name, options, expected_mode in [('420.jpg', {'subsampling': 2}, 1),
                                                     ('444.jpg', {'subsampling': 0}, 0),
                                                     ('image.png', {}, 0)]:
                    path = Path(directory) / name
                    Image.fromarray(pixels).save(path, **options)
                    item = dict(file_name=name, height=35, width=47)
                    header = np.zeros(4, np.int32)
                    storage = np.full(18 * 24 * 3 + 16, 177, np.uint8)
                    decoder(item, directory, storage[:-16].reshape(18, 24, 3), 2, header)
                    self.assertEqual(header[3], expected_mode)
                    self.assertTrue((storage[-16:] == 177).all())
                self.assertTrue(fallback.called)
                # Oversized-image resize uses the established RGB path.
                header = np.zeros(4, np.int32)
                decoder(dict(file_name='420.jpg', height=35, width=47), directory,
                        np.empty((12, 12, 3), np.uint8), 2, header)
                self.assertEqual(header[3], 0)
                decoder(dict(file_name='image.png', height=35, width=47), directory,
                        np.empty((12, 12, 3), np.uint8), 2, header)
        finally:
            decoder.close()

    @unittest.skipUnless((ROOT / 'weights/trt_yuv.ptx').exists(), 'requires local CUDA artifacts')
    def test_cuda_mixed_formats_odd_dimensions_and_tail(self):
        from cuda.bindings import driver as cu
        if int(cu.cuInit(0)[0]):
            self.skipTest('CUDA unavailable')
        context = check(cu.cuDevicePrimaryCtxRetain(0))
        check(cu.cuCtxSetCurrent(context))
        side, batch = 54, 3
        h, w = 35, 47
        rng = np.random.default_rng(7)
        y = rng.integers(0, 256, (36, 48), np.uint8)
        cb = rng.integers(0, 256, (18, 24), np.uint8)
        cr = rng.integers(0, 256, (18, 24), np.uint8)
        packed_yuv = np.concatenate([a.ravel() for a in (y, cb, cr)])
        yy = y[:h, :w].astype(np.int32)
        u = cb.repeat(2, 0).repeat(2, 1)[:h, :w].astype(np.int32) - 128
        v = cr.repeat(2, 0).repeat(2, 1)[:h, :w].astype(np.int32) - 128
        rgb = np.stack([yy + ((91881*v+32768)>>16), yy + ((-22554*u-46802*v+32768)>>16),
                        yy + ((116130*u+32768)>>16)], -1).clip(0, 255).astype(np.uint8)
        packed = np.zeros(batch * 16 + packed_yuv.size + rgb.size, np.uint8)
        header = packed[:batch*16].view(np.int32).reshape(batch, 4)
        header[0] = [w, h, batch*16, 1]
        header[1] = [w, h, batch*16+packed_yuv.size, 0]
        packed[batch*16:batch*16+packed_yuv.size] = packed_yuv
        packed[batch*16+packed_yuv.size:] = rgb.ravel()
        actual = np.empty((batch, 3, side, side), np.float32)
        source = check(cu.cuMemAlloc(packed.nbytes))
        target = check(cu.cuMemAlloc(actual.nbytes))
        module = check(cu.cuModuleLoadData((ROOT/'weights/trt_yuv.ptx').read_bytes()))
        try:
            check(cu.cuMemcpyHtoD(source, packed.ctypes.data, packed.nbytes))
            kernel = check(cu.cuModuleGetFunction(module, b'normalize'))
            args = [ctypes.c_uint64(int(source)), ctypes.c_uint64(int(target)), ctypes.c_int(side), ctypes.c_int(batch)]
            params = np.array([ctypes.addressof(a) for a in args], np.uint64)
            check(cu.cuLaunchKernel(kernel, (actual.size+255)//256, 1, 1, 256, 1, 1, 0, 0, params.ctypes.data, 0))
            check(cu.cuMemcpyDtoH(actual.ctypes.data, target, actual.nbytes))
            np.testing.assert_array_equal(actual[0], actual[1])
            expected = cv2.resize(rgb, (side, side)).transpose(2, 0, 1)/255.
            np.testing.assert_allclose(actual[0], expected, atol=1.01/255, rtol=0)
            self.assertFalse(actual[2].any())
        finally:
            check(cu.cuMemFree(source))
            check(cu.cuMemFree(target))
            check(cu.cuModuleUnload(module))
            check(cu.cuDevicePrimaryCtxRelease(0))
