"""Copy public images into a reproducible throughput workload, without labels.

Repeated public images are useful for testing pipeline duration and memory use,
but cannot represent hidden-test accuracy or its image/content distribution.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/SuperShip/throughput4300'))
    parser.add_argument('--count', type=int, default=4300)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    sources = []
    categories = None
    for split, metadata in (
        ('val', 'data/SuperShip/val/annotations/instances_val.json'),
        ('reference', 'data/SuperShip/reference/reference_images_meta.json'),
    ):
        content = json.loads(Path(metadata).read_text())
        if categories is None:
            categories = content['categories']
        for item in content['images']:
            path = Path('data/SuperShip') / split / 'images' / item['file_name']
            if not path.is_file():
                raise FileNotFoundError(path)
            sources.append((split, item, path))
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    chosen = [sources[index % len(sources)] for index in range(args.count)]
    rng.shuffle(chosen)
    # Refuse an existing destination rather than mixing old and new workloads.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = args.output_dir / 'images'
    image_dir.mkdir()
    images, provenance = [], []
    for image_id, (split, item, path) in enumerate(chosen, 1):
        name = f'{image_id:06d}{path.suffix}'
        # Independent file copies, not symlinks/hardlinks sharing page-cache entries.
        shutil.copyfile(path, image_dir / name)
        images.append(dict(id=image_id, file_name=name, width=item['width'], height=item['height']))
        provenance.append(dict(id=image_id, source_split=split, source_id=item['id'], source_file=str(path)))
    metadata = dict(images=images, categories=categories)
    (args.output_dir / 'images_meta.json').write_text(json.dumps(metadata) + '\n')
    manifest = dict(
        purpose='synthetic throughput only; do not use for accuracy', seed=args.seed,
        count=args.count, unique_source_images=len({str(path) for _, _, path in chosen}),
        split_counts=dict(Counter(split for split, _, _ in chosen)), provenance=provenance,
    )
    (args.output_dir / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'provenance'}))
    print(f'Images: {image_dir}\nMetadata: {args.output_dir / "images_meta.json"}')


if __name__ == '__main__':
    main()
