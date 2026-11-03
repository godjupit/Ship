"""Shared validation of competition metadata and prediction files."""

import json
import math


def reject_json_constant(value):
    raise ValueError(f"JSON contains a non-finite number: {value}")


def finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        reject_json_constant(value)
    return number


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(
            source, parse_constant=reject_json_constant,
            parse_float=finite_json_float, object_pairs_hook=unique_json_object,
        )


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def read_submission(path, meta_file=None):
    submission = load_json(path)
    if not isinstance(submission, list):
        raise ValueError("submission must be a JSON array")

    image_ids = None
    if meta_file is not None:
        metadata = load_json(meta_file)
        if not isinstance(metadata, dict) or not isinstance(metadata.get("images"), list):
            raise ValueError("metadata must contain an images array")
        image_ids = set()
        for image in metadata["images"]:
            if not isinstance(image, dict) or type(image.get("id")) is not int:
                raise ValueError("metadata contains an invalid image id")
            if image["id"] in image_ids:
                raise ValueError("metadata contains duplicate image ids")
            image_ids.add(image["id"])

    required = {"image_id", "category_id", "bbox", "score"}
    for index, detection in enumerate(submission):
        if not isinstance(detection, dict) or not required.issubset(detection):
            raise ValueError(f"detection {index} has missing or invalid fields")
        if type(detection["image_id"]) is not int:
            raise ValueError(f"detection {index} has an invalid image_id")
        if image_ids is not None and detection["image_id"] not in image_ids:
            raise ValueError(f"detection {index} has an unknown image_id")
        category = detection["category_id"]
        if type(category) is not int or category not in (0, 1, 2):
            raise ValueError(f"detection {index} has an invalid category_id")
        bbox = detection["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"detection {index} has an invalid bbox")
        if not all(finite_number(value) for value in [*bbox, detection["score"]]):
            raise ValueError(f"detection {index} contains an invalid number")
        if not 0 <= detection["score"] <= 1:
            raise ValueError(f"detection {index} has an invalid score")
        # Non-positive dimensions are scored as unmatched detections, not a
        # malformed submission. Valid geometry must remain representable.
        if bbox[2] > 0 and bbox[3] > 0:
            if not all(finite_number(value) for value in (
                bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[2] * bbox[3],
            )):
                raise ValueError(f"detection {index} has overflowing geometry")
    return submission


def validate_output(path, meta_file=None):
    return len(read_submission(path, meta_file))
