"""Reuse TurboJPEG handles and decode directly into bounded pinned RGB/YUV slots."""
import ctypes as c
from pathlib import Path
import threading


def has_app1(data):
    """Inspect JPEG header markers only; do not scan entropy-coded image bytes."""
    offset = 2
    while offset + 3 < len(data):
        if data[offset] != 255:
            return True  # Malformed headers use the established decoder.
        while offset < len(data) and data[offset] == 255:
            offset += 1
        if offset >= len(data):
            return True
        marker = data[offset]
        offset += 1
        if marker in (0xDA, 0xD9):
            return False
        if marker == 0xE1:
            return True
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            continue
        if offset + 1 >= len(data):
            return True
        length = data[offset] * 256 + data[offset + 1]
        if length < 2:
            return True
        offset += length
    return True


class JPEGDecoder:
    def __init__(self, fallback, mode):
        import turbojpeg

        library = next((Path(turbojpeg.__file__).parent / 'turbojpeg.libs').glob('libturbojpeg*.so*'))
        self.lib = c.CDLL(str(library))
        self.lib.tjInitDecompress.restype = c.c_void_p
        self.lib.tjDestroy.argtypes = [c.c_void_p]
        self.lib.tjDecompressHeader3.argtypes = [c.c_void_p, c.c_char_p, c.c_ulong,
                                                *([c.POINTER(c.c_int)] * 4)]
        self.lib.tjDecompress2.argtypes = [c.c_void_p, c.c_char_p, c.c_ulong, c.c_void_p,
                                         c.c_int, c.c_int, c.c_int, c.c_int, c.c_int]
        self.lib.tjDecompressToYUV2.argtypes = [c.c_void_p, c.c_char_p, c.c_ulong, c.c_void_p,
                                              c.c_int, c.c_int, c.c_int, c.c_int]
        self.local = threading.local()
        self.handles = []
        self.lock = threading.Lock()
        self.fallback, self.mode = fallback, mode

    def close(self):
        """Call only after all decoder jobs have finished."""
        for handle in self.handles:
            self.lib.tjDestroy(handle)
        self.handles.clear()
        self.local = threading.local()

    def __call__(self, item, directory, destination, reduction, descriptor):
        """Return RGB for unsupported formats; descriptor 1 marks planar 4:2:0."""
        path = Path(directory) / item['file_name']
        data = path.read_bytes()
        expected = tuple((item[k] + reduction - 1) // reduction for k in ('height', 'width'))
        # Preserve OpenCV's orientation handling and oversized-image fallback.
        if (not data.startswith(b'\xff\xd8') or has_app1(data)
                or destination.shape[:2] != expected):
            return self.fallback(item, directory, destination, reduction)
        if not hasattr(self.local, 'handle'):
            handle = self.lib.tjInitDecompress()
            if not handle:
                raise RuntimeError('cannot create JPEG decoder')
            self.local.handle = handle
            with self.lock:
                self.handles.append(handle)
        handle = self.local.handle
        width, height, sampling, color = (c.c_int() for _ in range(4))
        if self.lib.tjDecompressHeader3(handle, data, len(data), c.byref(width), c.byref(height),
                                       c.byref(sampling), c.byref(color)):
            return self.fallback(item, directory, destination, reduction)
        if (width.value, height.value) != (item['width'], item['height']):
            raise ValueError(f'metadata dimensions disagree with decoded image: {path}')
        h, w = expected
        yuv_bytes = ((w + 1) // 2) * ((h + 1) // 2) * 6
        use_yuv = (self.mode == 'yuv' and sampling.value == 2 and color.value == 1
                   and yuv_bytes <= destination.nbytes)
        if use_yuv:
            result = self.lib.tjDecompressToYUV2(handle, data, len(data), destination.ctypes.data,
                                                w, 1, h, 0)
        elif color.value in (0, 1, 2):
            result = self.lib.tjDecompress2(handle, data, len(data), destination.ctypes.data,
                                           w, destination.strides[0], h, 0, 0)
        else:
            return self.fallback(item, directory, destination, reduction)
        if result:
            return self.fallback(item, directory, destination, reduction)
        descriptor[3] = int(use_yuv)
