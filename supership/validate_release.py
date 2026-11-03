"""Validate the public SuperShip code repository and installed dataset."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SuperShip"


def load_json(path):
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def check_split(name, annotation_path, image_dir, expected_images, expected_annotations):
    data = load_json(annotation_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    assert len(images) == expected_images, (name, len(images), expected_images)
    assert len(annotations) == expected_annotations, (
        name,
        len(annotations),
        expected_annotations,
    )
    missing = [item["file_name"] for item in images if not (image_dir / item["file_name"]).is_file()]
    assert not missing, f"{name}: missing images: {missing[:5]}"
    actual_files = {path.name for path in image_dir.iterdir() if path.is_file()}
    expected_files = {item["file_name"] for item in images}
    assert actual_files == expected_files, (
        f"{name}: image directory differs from metadata; "
        f"extra={sorted(actual_files - expected_files)[:5]}, "
        f"missing={sorted(expected_files - actual_files)[:5]}"
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_private_boundary():
    forbidden = {
        "test_gt.json",
        "test_images_meta.json",
        "instances_test.json",
    }
    leaked = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.name in forbidden]
    assert not leaked, f"private test files found: {leaked}"
    assert not (ROOT / "organizer").exists(), "organizer-only directory must not be public"


def check_dataset_checksums():
    checksum_file = DATA / "SHA256SUMS"
    assert checksum_file.is_file(), f"missing dataset checksum file: {checksum_file}"
    for line_number, line in enumerate(checksum_file.read_text(encoding="utf-8").splitlines(), 1):
        expected, relative_path = line.split("  ", 1)
        path = ROOT / relative_path
        assert path.is_file(), f"checksum entry {line_number} is missing: {relative_path}"
        assert sha256(path) == expected, f"checksum mismatch: {relative_path}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="check repository/private boundary without requiring the cloud dataset",
    )
    args = parser.parse_args()

    check_private_boundary()
    release_info = load_json(ROOT / "RELEASE_INFO.json")
    assert sha256(ROOT / release_info["baseline_weight"]) == release_info["baseline_weight_sha256"]
    assert release_info["splits"]["hidden"] == {"images": 4300, "included": False}

    if args.code_only:
        assert not (DATA / "train").exists(), "public data should not be committed to the code repository"
        print("PASS: SuperShip code release contains no known private-test or cloud-dataset files")
        return

    if not (DATA / "train").is_dir():
        raise SystemExit(
            "SuperShip public data is not installed. Run download_data.sh first, "
            "or use --code-only when checking the repository before publication."
        )

    check_split(
        "train",
        DATA / "train/annotations/instances_train.json",
        DATA / "train/images",
        2405,
        11904,
    )
    check_split(
        "val",
        DATA / "val/annotations/instances_val.json",
        DATA / "val/images",
        1030,
        4576,
    )
    reference_gt = DATA / "reference/annotations/instances_reference.json"
    check_split(
        "reference",
        reference_gt,
        DATA / "reference/images",
        200,
        519,
    )

    reference_meta = DATA / "reference/reference_images_meta.json"
    metadata = load_json(reference_meta)
    assert "annotations" not in metadata, "reference inference metadata leaks annotations"
    assert len(metadata.get("images", [])) == 200
    assert sorted(item["id"] for item in metadata["images"]) == list(range(1, 201))

    manifest = load_json(DATA / "reference/split_manifest.json")
    assert sha256(reference_meta) == manifest["meta_sha256"]
    assert sha256(reference_gt) == manifest["annotation_sha256"]
    check_dataset_checksums()

    print("PASS: SuperShip public release is complete and contains no known private-test files")


if __name__ == "__main__":
    main()
