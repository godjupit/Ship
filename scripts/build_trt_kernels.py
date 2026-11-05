"""Compile standalone TensorRT preprocessing kernels for an offline runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='weights/trt_resize.ptx')
    parser.add_argument('--nvcc', default=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'supership/trt_resize.cu'
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.nvcc, '-ptx', '-arch=compute_89', str(source), '-o', str(output)], check=True)
    manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    ptx_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), arch='compute_89')
    Path(str(output) + '.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Built {output}')


if __name__ == '__main__':
    main()
