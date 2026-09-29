"""
Fix C, stage 2 groundwork: crop out every labeled chromosome from the
dataset and organize the crops into a classification-ready folder
structure (one subfolder per class), so a small dedicated classifier can
be trained to answer "which of the 24 types is this one chromosome" - a
much narrower, easier task than full detection.

No new labeling work needed: this crops directly from the same VOC XML
annotations voc_to_yolo.py already parses, reusing the same official
train/val split so nothing leaks between the two.
"""

import glob
import os

from PIL import Image

from voc_to_yolo import (
    CLASSES, IMG_ROOT, TEST_TXT, TRAIN_TXT, XML_ROOT,
    index_images, load_split_file, parse_annotation,
)

CROPS_ROOT = r"C:\Users\Phili\OneDrive\Documents\Projects\Gene Mutation\raw\chromosome_crops"

# A little padding around each box so the classifier sees some context
# rather than a razor-tight crop that might clip an edge of the chromosome.
PADDING = 4


def crop_box(image, xmin, ymin, xmax, ymax, padding=PADDING):
    w, h = image.size
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(w, xmax + padding)
    ymax = min(h, ymax + padding)
    return image.crop((xmin, ymin, xmax, ymax))


def main():
    xml_files = glob.glob(os.path.join(XML_ROOT, "**", "*.xml"), recursive=True)
    print(f"XML files found: {len(xml_files)}")

    image_index = index_images(IMG_ROOT)
    train_stems = load_split_file(TRAIN_TXT)
    test_stems = load_split_file(TEST_TXT)
    use_official_split = bool(train_stems and test_stems)
    print(f"Using official split: {use_official_split}")

    for split in ("train", "val"):
        for name in CLASSES:
            os.makedirs(os.path.join(CROPS_ROOT, split, name), exist_ok=True)

    counts = {"train": 0, "val": 0}
    skipped_no_image = 0
    skipped_no_split = 0

    for xml_path in xml_files:
        stem = os.path.splitext(os.path.basename(xml_path))[0]

        if use_official_split:
            if stem in train_stems:
                split = "train"
            elif stem in test_stems:
                split = "val"
            else:
                skipped_no_split += 1
                continue
        else:
            split = "train"

        img_path = image_index.get(stem)
        if img_path is None:
            skipped_no_image += 1
            continue

        parsed = parse_annotation(xml_path)
        if parsed is None:
            continue
        width, height, rows = parsed
        if not rows:
            continue

        image = Image.open(img_path).convert("RGB")

        for idx, (cls_id, x, y, w, h) in enumerate(rows):
            xmin = (x - w / 2) * width
            ymin = (y - h / 2) * height
            xmax = (x + w / 2) * width
            ymax = (y + h / 2) * height
            crop = crop_box(image, xmin, ymin, xmax, ymax)

            name = CLASSES[cls_id]
            out_path = os.path.join(CROPS_ROOT, split, name, f"{stem}_{idx}.jpg")
            crop.save(out_path)
            counts[split] += 1

    print(f"\ntrain crops: {counts['train']}")
    print(f"val crops:   {counts['val']}")
    if skipped_no_image:
        print(f"skipped (no matching image): {skipped_no_image}")
    if skipped_no_split:
        print(f"skipped (not in either split file): {skipped_no_split}")


if __name__ == "__main__":
    main()
