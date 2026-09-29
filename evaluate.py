"""
Run the full pipeline over every validation image and score it against the
clinical ground truth, per image and per patient.

    python evaluate.py
    python evaluate.py --no-case-aggregation     # judge each image alone

Pipeline: detect -> merge duplicate boxes -> classify crops -> assign crops
to the 46 karyotype slots -> flag count mismatches -> confirm across the
patient's images.
"""

import determinism  # noqa: F401 (import first - must precede any torch/ultralytics import)

import argparse
import csv
import glob
import os
from collections import Counter

from ultralytics import YOLO

from assignment_solve import diagnose_image, get_crop_probs
from case_aggregate import NEEDS_MORE_CELLS, aggregate_by_case, direction, format_case_report, score_by_case
from config import AGNOSTIC_IOU, CLASSIFIER, CONF, DETECTOR, IMGSZ, IOU, NO_MATCH_PROB
from detect import detect
from ground_truth import load_ground_truth
from scoring import format_report, score
from voc_to_yolo import OUT_ROOT


def run_inference(detector_weights=DETECTOR, classifier_weights=CLASSIFIER, device="0", conf=CONF,
                  iou=IOU, imgsz=IMGSZ, agnostic_iou=AGNOSTIC_IOU, no_match_prob=NO_MATCH_PROB, limit=None):
    """Per-image diagnosis rows for the validation images, paired with ground truth."""
    detector, classifier = YOLO(detector_weights), YOLO(classifier_weights)
    truth = load_ground_truth()
    val_images = sorted(glob.glob(os.path.join(OUT_ROOT, "images", "val", "*")))[:limit]

    rows = []
    for image_path in val_images:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        gt = truth.get(stem)
        if gt is None:
            continue
        detections = detect(detector, image_path, device=device, conf=conf, iou=iou,
                            imgsz=imgsz, agnostic_iou=agnostic_iou)
        result = diagnose_image(get_crop_probs(classifier, image_path, detections, device=device),
                                no_match_prob=no_match_prob)
        rows.append({
            "stem": stem,
            "gt_verdict": gt.verdict,
            "gt_kind": gt.kind or "normal",
            "gt_chromosomes": ";".join(gt.chromosomes),
            "pred_verdict": result.verdict,
            "pred_chromosomes": ";".join(a.chromosome for a in result.abnormalities),
            "pred_direction": direction(result.abnormalities),
            "pred_total": result.total,
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", default=DETECTOR)
    parser.add_argument("--classifier", default=CLASSIFIER)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--iou", type=float, default=IOU)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--agnostic-iou", type=float, default=AGNOSTIC_IOU,
                        help="merge boxes overlapping more than this regardless of label (1.0 disables)")
    parser.add_argument("--no-match-prob", type=float, default=NO_MATCH_PROB)
    parser.add_argument("--no-case-aggregation", action="store_true",
                        help="score each image on its own instead of confirming findings "
                             "across images of the same patient")
    parser.add_argument("--limit", type=int, default=None, help="only the first N val images")
    parser.add_argument("--csv-out", default="runs/evaluate_results.csv")
    args = parser.parse_args()

    rows = run_inference(args.detector, args.classifier, device=args.device, conf=args.conf,
                         iou=args.iou, imgsz=args.imgsz, agnostic_iou=args.agnostic_iou,
                         no_match_prob=args.no_match_prob, limit=args.limit)
    if not args.no_case_aggregation:
        rows = aggregate_by_case(rows)

    print(format_report(score(rows)))
    held = Counter(r["gt_kind"] for r in rows if r["pred_verdict"] == NEEDS_MORE_CELLS)
    if held:
        print(f"\nNeeds more cells (not counted as flagged above): {sum(held.values())} images, "
              f"by ground truth: {dict(held)}")
    print(format_case_report(score_by_case(rows)))

    if rows:
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-image results saved to: {args.csv_out}")
