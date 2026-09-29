"""
Build the classifier's training data from the DETECTOR's own predicted
boxes rather than the hand-drawn ones, so the classifier learns the box
style it actually sees in the pipeline. (A classifier trained on hand-drawn
crops became badly overconfident on detector crops.)

Each predicted box takes the label of the ground-truth box it overlaps
most; boxes with no confident match (spurious or duplicate detections) are
skipped. The classifier in config.CLASSIFIER (train-5) was trained on a
class-balanced subset of this output (1,500 crops per class for train, 400
for val), built by hand and kept in raw/chromosome_crops_from_preds_subset.
"""

import glob
import os

from PIL import Image
from ultralytics import YOLO

from config import DETECTOR
from detect import box_iou, predict_image
from extract_crops import crop_box
from voc_to_yolo import (
    CLASSES, IMG_ROOT, TEST_TXT, TRAIN_TXT, XML_ROOT,
    index_images, load_split_file, parse_annotation,
)

CROPS_ROOT = r"C:\Users\Phili\OneDrive\Documents\Projects\Gene Mutation\raw\chromosome_crops_from_preds"

# Standard object-detection convention: a predicted box "is" a ground-truth
# object once they overlap at least this much.
MATCH_IOU_THRESHOLD = 0.5


def ground_truth_boxes(xml_path):
    """Return [(class_name, xmin, ymin, xmax, ymax), ...] in pixel
    coordinates for one image's real annotations."""
    parsed = parse_annotation(xml_path)
    if parsed is None:
        return []
    width, height, rows = parsed
    boxes = []
    for cls_id, x, y, w, h in rows:
        xmin = (x - w / 2) * width
        ymin = (y - h / 2) * height
        xmax = (x + w / 2) * width
        ymax = (y + h / 2) * height
        boxes.append((CLASSES[cls_id], xmin, ymin, xmax, ymax))
    return boxes


def main():
    model = YOLO(DETECTOR)
    xml_files = glob.glob(os.path.join(XML_ROOT, "**", "*.xml"), recursive=True)
    print(f"XML files found: {len(xml_files)}")

    image_index = index_images(IMG_ROOT)
    train_stems = load_split_file(TRAIN_TXT)
    test_stems = load_split_file(TEST_TXT)

    for split in ("train", "val"):
        for name in CLASSES:
            os.makedirs(os.path.join(CROPS_ROOT, split, name), exist_ok=True)

    counts = {"train": 0, "val": 0}
    unmatched = 0

    for i, xml_path in enumerate(xml_files):
        stem = os.path.splitext(os.path.basename(xml_path))[0]
        if stem in train_stems:
            split = "train"
        elif stem in test_stems:
            split = "val"
        else:
            continue

        img_path = image_index.get(stem)
        if img_path is None:
            continue

        gt_boxes = ground_truth_boxes(xml_path)
        if not gt_boxes:
            continue

        preds = predict_image(model, img_path)

        image = Image.open(img_path).convert("RGB")
        for idx, pred in enumerate(preds):
            best_iou, best_cls = 0.0, None
            for cls_name, gx1, gy1, gx2, gy2 in gt_boxes:
                iou = box_iou(pred["bbox"], (gx1, gy1, gx2, gy2))
                if iou > best_iou:
                    best_iou, best_cls = iou, cls_name
            if best_iou < MATCH_IOU_THRESHOLD:
                unmatched += 1
                continue

            crop = crop_box(image, *pred["bbox"])
            out_path = os.path.join(CROPS_ROOT, split, best_cls, f"{stem}_{idx}.jpg")
            crop.save(out_path)
            counts[split] += 1

        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(xml_files)} images "
                  f"(train={counts['train']} val={counts['val']} unmatched={unmatched})")

    print(f"\ntrain crops: {counts['train']}")
    print(f"val crops:   {counts['val']}")
    print(f"unmatched (skipped, no confident ground-truth match): {unmatched}")


if __name__ == "__main__":
    main()
