#!/usr/bin/env python3
"""Validate and summarize the CVPDL HW1 object-detection dataset.

The script reads the supplied COCO annotations and YOLO labels without
modifying them.  It writes a machine-readable summary plus plots that can be
used in the report and to choose training settings.

Example:
    python src/analyze_dataset.py \
        --data-dir dataset \
        --output-dir outputs/eda
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Any

# Keep Matplotlib's cache outside the dataset and avoid relying on a writable
# user-home directory when the script runs in a container or CI.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cvpdl_hw1_mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--samples-per-split", type=int, default=4)
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error("--samples-per-split must be positive")
    return args


def read_coco(data_dir: Path, split: str) -> dict[str, Any]:
    annotation_path = data_dir / "annotations" / f"instances_{split}.json"
    with annotation_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    annotations = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number} has {len(fields)} fields; expected 5")
        class_id, x_center, y_center, width, height = fields
        values = (int(class_id), float(x_center), float(y_center), float(width), float(height))
        if not all(0.0 <= value <= 1.0 for value in values[1:]):
            raise ValueError(f"{path}:{line_number} contains non-normalized coordinates: {values[1:]}")
        annotations.append(values)
    return annotations


def summarize_split(data_dir: Path, split: str, category_names: dict[int, str]) -> dict[str, Any]:
    coco = read_coco(data_dir, split)
    image_dir = data_dir / "images" / split
    label_dir = data_dir / "labels" / split

    images = coco["images"]
    annotations = coco["annotations"]
    image_ids = {image["id"] for image in images}
    image_paths = sorted(image_dir.glob("*.jpg"))
    label_paths = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}

    if image_stems != label_stems:
        raise ValueError(
            f"{split}: image/label IDs differ; "
            f"images_only={len(image_stems - label_stems)}, labels_only={len(label_stems - image_stems)}"
        )
    if image_stems != image_ids:
        raise ValueError(
            f"{split}: COCO image IDs do not match image filenames; "
            f"coco_only={len(image_ids - image_stems)}, files_only={len(image_stems - image_ids)}"
        )

    yolo_class_counts: collections.Counter[int] = collections.Counter()
    yolo_annotation_count = 0
    for label_path in label_paths:
        labels = read_yolo_label(label_path)
        yolo_annotation_count += len(labels)
        yolo_class_counts.update(label[0] for label in labels)

    coco_class_counts: collections.Counter[int] = collections.Counter(
        annotation["category_id"] for annotation in annotations
    )
    if yolo_annotation_count != len(annotations) or yolo_class_counts != coco_class_counts:
        raise ValueError(
            f"{split}: YOLO and COCO annotation totals/classes disagree: "
            f"yolo={yolo_annotation_count}, coco={len(annotations)}"
        )

    annotations_by_image: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation["image_id"]].append(annotation)

    box_areas = [annotation["area"] for annotation in annotations]
    box_counts = [len(annotations_by_image[image["id"]]) for image in images]
    widths = [image["width"] for image in images]
    heights = [image["height"] for image in images]
    size_bands = {
        "small_lt_32_squared": sum(area < 32**2 for area in box_areas),
        "medium_32_squared_to_96_squared": sum(32**2 <= area < 96**2 for area in box_areas),
        "large_ge_96_squared": sum(area >= 96**2 for area in box_areas),
    }

    return {
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "yolo_label_files": len(label_paths),
        "classes": {category_names[class_id]: coco_class_counts[class_id] for class_id in sorted(category_names)},
        "image_dimensions": {
            "width": {"min": min(widths), "max": max(widths), "mean": round(fmean(widths), 2)},
            "height": {"min": min(heights), "max": max(heights), "mean": round(fmean(heights), 2)},
            "unique_pairs": len({(image["width"], image["height"]) for image in images}),
        },
        "boxes_per_image": {
            "min": min(box_counts),
            "max": max(box_counts),
            "mean": round(fmean(box_counts), 2),
            "images_without_boxes": sum(count == 0 for count in box_counts),
        },
        "box_area_pixels": {
            "min": min(box_areas),
            "max": max(box_areas),
            "mean": round(fmean(box_areas), 2),
            **size_bands,
        },
        "occlusion": dict(sorted(collections.Counter(annotation.get("occlusion", 0) for annotation in annotations).items())),
        "truncation": dict(sorted(collections.Counter(annotation.get("truncation", 0) for annotation in annotations).items())),
        "_plot_data": {
            "box_counts": box_counts,
            "sqrt_areas": [math.sqrt(area) for area in box_areas],
            "class_counts": [coco_class_counts[class_id] for class_id in sorted(category_names)],
        },
        "_images": images,
        "_annotations_by_image": annotations_by_image,
    }


def plot_class_counts(train: dict[str, Any], val: dict[str, Any], category_names: dict[int, str], output_dir: Path) -> None:
    names = [category_names[index] for index in sorted(category_names)]
    positions = list(range(len(names)))
    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.bar([position - 0.2 for position in positions], train["_plot_data"]["class_counts"], 0.4, label="train")
    axis.bar([position + 0.2 for position in positions], val["_plot_data"]["class_counts"], 0.4, label="val")
    axis.set_xticks(positions, names, rotation=30, ha="right")
    axis.set_ylabel("annotations")
    axis.set_title("Class distribution")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close(fig)


def plot_density_and_size(train: dict[str, Any], val: dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for split_summary, color in ((train, PALETTE[0]), (val, PALETTE[1])):
        axes[0].hist(split_summary["_plot_data"]["box_counts"], bins=50, alpha=0.55, label=split_summary["split"], color=color)
        axes[1].hist(split_summary["_plot_data"]["sqrt_areas"], bins=60, alpha=0.55, label=split_summary["split"], color=color)
    axes[0].set_title("Object density per image")
    axes[0].set_xlabel("boxes per image")
    axes[0].set_ylabel("images")
    axes[1].set_title("Box scale distribution")
    axes[1].set_xlabel("sqrt(box area) in pixels")
    axes[1].set_ylabel("annotations")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output_dir / "density_and_box_scale.png", dpi=180)
    plt.close(fig)


def draw_samples(summary: dict[str, Any], data_dir: Path, output_dir: Path, category_names: dict[int, str], seed: int, samples: int) -> None:
    rng = random.Random(seed)
    selected = rng.sample(summary["_images"], k=min(samples, len(summary["_images"])))
    tiles: list[Image.Image] = []
    for image_info in selected:
        image = Image.open(data_dir / "images" / summary["split"] / image_info["file_name"]).convert("RGB")
        image.thumbnail((720, 540))
        scale_x = image.width / image_info["width"]
        scale_y = image.height / image_info["height"]
        draw = ImageDraw.Draw(image)
        for annotation in summary["_annotations_by_image"][image_info["id"]]:
            x, y, width, height = annotation["bbox"]
            box = (x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y)
            color = PALETTE[annotation["category_id"] % len(PALETTE)]
            draw.rectangle(box, outline=color, width=max(1, image.width // 450))
            draw.text((box[0], max(0, box[1] - 12)), category_names[annotation["category_id"]], fill=color)
        tiles.append(image)

    canvas_width = max(tile.width for tile in tiles)
    canvas_height = max(tile.height for tile in tiles)
    columns = min(2, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    grid = Image.new("RGB", (canvas_width * columns, canvas_height * rows), "white")
    for index, tile in enumerate(tiles):
        grid.paste(tile, ((index % columns) * canvas_width, (index // columns) * canvas_height))
    grid.save(output_dir / f"samples_{summary['split']}.jpg", quality=94)


def strip_plot_data(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    classes_path = data_dir / "classes.txt"
    category_names = {index: name for index, name in enumerate(classes_path.read_text(encoding="utf-8").splitlines()) if name}
    if len(category_names) != 10:
        raise ValueError(f"Expected 10 classes in {classes_path}, found {len(category_names)}")

    train = summarize_split(data_dir, "train", category_names)
    val = summarize_split(data_dir, "val", category_names)
    sample_submission = data_dir / "sample_submission.csv"
    test_images = sorted((data_dir / "images" / "test").glob("*.jpg"))
    submission_rows = len(sample_submission.read_text(encoding="utf-8").splitlines()) - 1
    if len(test_images) != submission_rows:
        raise ValueError(f"Test image count ({len(test_images)}) differs from submission rows ({submission_rows})")

    plot_class_counts(train, val, category_names, output_dir)
    plot_density_and_size(train, val, output_dir)
    draw_samples(train, data_dir, output_dir, category_names, args.seed, args.samples_per_split)
    draw_samples(val, data_dir, output_dir, category_names, args.seed + 1, args.samples_per_split)

    report = {
        "data_dir": str(data_dir),
        "classes": category_names,
        "train": strip_plot_data(train),
        "val": strip_plot_data(val),
        "test": {"images": len(test_images), "sample_submission_rows": submission_rows},
        "validation": "COCO image IDs, image files, YOLO label files, and per-class annotation totals match for train and val.",
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote plots and summary to {output_dir}")


if __name__ == "__main__":
    main()
