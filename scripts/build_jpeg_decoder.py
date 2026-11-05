"""Build the measured libjpeg-turbo 3.0.0 / 9-bit / Zen 2 PGO decoder offline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REVISION = '6c87537f60941f3c265c339fe60d1e31d2a42ccf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--training-dir', type=Path, default=Path('data/SuperShip/train/images'))
    parser.add_argument('--work-dir', type=Path, default=Path('.local-tools/jpeg-pgo'))
    parser.add_argument('--output', type=Path, default=Path('weights/libturbojpeg-pgo.so'))
    parser.add_argument('--nasm', default=shutil.which('nasm'))
    args = parser.parse_args()
    source, work, output = args.source.resolve(), args.work_dir.resolve(), args.output.resolve()
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != REVISION:
        raise ValueError('source must be the pinned libjpeg-turbo 3.0.0 revision')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--'], check=True,
                   stdout=subprocess.DEVNULL)
    if not args.nasm:
        raise ValueError('NASM is required to preserve SIMD decoding')
    # A fresh build directory avoids stale PGO profiles and source mutations.
    work.mkdir(parents=True, exist_ok=False)
    src, build, profiles = work / 'source', work / 'build', work / 'profiles'
    shutil.copytree(source, src, ignore=shutil.ignore_patterns('.git'))
    header = src / 'jdhuff.h'
    text = header.read_text()
    assert text.count('#define HUFF_LOOKAHEAD  8 ') == 1
    header.write_text(text.replace('#define HUFF_LOOKAHEAD  8 ', '#define HUFF_LOOKAHEAD  9 '))
    images = sorted(args.training_dir.resolve().rglob('*.jpg'))
    if not images:
        raise ValueError('public JPEG training images are required for offline PGO')
    manifest = work / 'training.txt'
    manifest.write_text(''.join(str(path) + '\n' for path in images))
    dataset_hash = hashlib.sha256()
    for path in images:
        dataset_hash.update(hashlib.sha256(path.read_bytes()).digest())
    flags = '-O3 -DNDEBUG -march=znver2 -mtune=znver2'
    trainer = work / 'train'
    trainer_source = Path(__file__).with_name('experiments') / 'train_jpeg_pgo.cpp'
    with (work / 'build.log').open('w') as log:
        def run(command):
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

        run(['g++', '-O3', '-std=c++17', '-pthread', str(trainer_source), '-ldl', '-o', str(trainer)])
        for phase in ('generate', 'use'):
            profile_flags = (f'-fprofile-generate={profiles} -fprofile-update=atomic' if phase == 'generate'
                             else f'-fprofile-use={profiles} -fprofile-correction -Werror=missing-profile')
            run(['cmake', '-S', str(src), '-B', str(build), '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
                 '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_C_FLAGS_RELEASE={flags} {profile_flags}',
                 '-DENABLE_STATIC=OFF', '-DWITH_SIMD=ON', f'-DCMAKE_ASM_NASM_COMPILER={Path(args.nasm).resolve()}'])
            run(['cmake', '--build', str(build), '--target', 'turbojpeg', '-j8'])
            if phase == 'generate':
                run([str(trainer), str(build / 'libturbojpeg.so'), str(manifest), 'yuv'])
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(build / 'libturbojpeg.so', output)
    metadata = dict(library_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    source_revision=REVISION, lookahead=9, flags=flags, training_images=len(images),
                    training_sha256=dataset_hash.hexdigest(),
                    compiler=subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0])
    Path(str(output) + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for name in ('LICENSE.md', 'README.ijg'):
        shutil.copyfile(source / name, Path(str(output) + '.' + name))
    print(f'Built {output}; details in {work / "build.log"}')


if __name__ == '__main__':
    main()
