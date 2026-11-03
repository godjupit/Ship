"""Visualize COCO-format annotations on an image.

Draws bboxes and category names (plus score if present) over the image and
saves the result.

Usage (from anywhere):
  python supership/visualize_annotations.py \
    --img data/SuperShip/train/images/000006.jpg \
    --ann data/SuperShip/train/annotations/instances_train.json \
    -o vis.jpg

The annotation json must be COCO style (bbox = [x, y, w, h] in pixels).
The image is located inside the json by file_name (basename match); for
submission-style jsons (a plain list without 'images'), pass --image-id.
Works for GT jsons and prediction jsons alike (score is shown when present).
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

# fixed color per category id (R, G, B)
PALETTE = [(230, 57, 70), (29, 53, 87), (251, 133, 0), (42, 157, 143),
           (114, 9, 183), (233, 196, 106), (0, 150, 199), (188, 71, 73)]


def load_font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def find_image_entry(ann_data, img_path, image_id):
    images = ann_data.get('images') if isinstance(ann_data, dict) else None
    if images is None:
        if image_id is None:
            raise SystemExit('--image-id is required for prediction arrays')
        return image_id
    basename = os.path.basename(img_path)
    for im in images:
        if im.get('file_name') == basename or \
                im.get('file_name', '').endswith('/' + basename):
            return im['id']
    if image_id is not None:
        return image_id
    raise SystemExit(f'image "{basename}" not found in annotation json; '
                     f'pass --image-id explicitly')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='image path')
    parser.add_argument('--ann', required=True, help='COCO annotation json')
    parser.add_argument('-o', '--output', default=None,
                        help='output path (default: <img>_vis.jpg)')
    parser.add_argument('--image-id', type=int, default=None,
                        help='image_id, for jsons without images index')
    parser.add_argument('--min-score', type=float, default=0.0,
                        help='skip boxes whose score < this value')
    parser.add_argument('--line-width', type=int, default=0,
                        help='bbox line width (default: auto by image size)')
    args = parser.parse_args()

    with open(args.ann) as f:
        ann_data = json.load(f)

    categories = {c['id']: c.get('name', str(c['id']))
                  for c in ann_data.get('categories', [])} if isinstance(ann_data, dict) else {}

    img_id = find_image_entry(ann_data, args.img, args.image_id)
    annotations = ann_data if isinstance(ann_data, list) else ann_data.get('annotations', [])
    anns = [a for a in annotations
            if a.get('image_id') == img_id and
            a.get('score', 1.0) >= args.min_score]

    img = Image.open(args.img).convert('RGB')
    draw = ImageDraw.Draw(img)
    lw = args.line_width or max(2, round(min(img.size) / 400))
    font = load_font(max(14, round(min(img.size) / 40)))

    n_drawn = 0
    for a in anns:
        x, y, w, h = a['bbox']
        x1, y1 = max(0, x), max(0, y)
        x2 = min(img.size[0], x + w)
        y2 = min(img.size[1], y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        cid = a['category_id']
        color = PALETTE[cid % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        label = categories.get(cid, str(cid))
        if 'score' in a:
            label += f' {a["score"]:.2f}'
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 4 if y1 - th - 4 >= 0 else y1 + 2
        draw.rectangle([x1, ty - 2, x1 + tw + 6, ty + th + 2], fill=color)
        draw.text((x1 + 3, ty), label, fill=(255, 255, 255), font=font)
        n_drawn += 1

    out = args.output or os.path.splitext(args.img)[0] + '_vis.jpg'
    img.save(out, quality=95)
    print(f'drawn {n_drawn} boxes (image_id={img_id}) -> {out}')


if __name__ == '__main__':
    main()
