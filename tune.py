"""
Sweep the pipeline's two tuned settings (AGNOSTIC_IOU and NO_MATCH_PROB)
cheaply. Detection and classification run once and are cached (boxes,
detector confidences, crop probabilities); each setting is then re-applied
from the cache and scored exactly like evaluate.py, including the patient-level
step.

    python tune.py            # delete runs/tune_cache.pkl after retraining a model
"""

import determinism  # noqa: F401 (import first - must precede any torch/ultralytics import)

import glob
import os
import pickle

from ultralytics import YOLO

from assignment_solve import diagnose_image, get_crop_probs
from case_aggregate import aggregate_by_case, direction
from config import AGNOSTIC_IOU, CLASSIFIER, DETECTOR, NO_MATCH_PROB
from detect import agnostic_keep, predict_image
from ground_truth import load_ground_truth
from scoring import score
from voc_to_yolo import OUT_ROOT

CACHE_PATH = "runs/tune_cache.pkl"


def build_cache():
    detector, classifier = YOLO(DETECTOR), YOLO(CLASSIFIER)
    truth = load_ground_truth()
    val_images = sorted(glob.glob(os.path.join(OUT_ROOT, "images", "val", "*")))

    cache = []
    for i, image_path in enumerate(val_images):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        gt = truth.get(stem)
        if gt is None:
            continue
        dets = predict_image(detector, image_path)
        probs = get_crop_probs(classifier, image_path, dets)
        cache.append((stem, gt, [d["bbox"] for d in dets], [d["confidence"] for d in dets], probs))
        if (i + 1) % 200 == 0:
            print(f"  cached {i + 1}/{len(val_images)}", flush=True)

    with open(CACHE_PATH, "wb") as fh:
        pickle.dump(cache, fh)
    return cache


def load_cache():
    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "rb") as fh:
            return pickle.load(fh)
    print("building cache (detector + classifier over all val images)...", flush=True)
    return build_cache()


def evaluate_settings(cache, agnostic_iou=AGNOSTIC_IOU, no_match_prob=NO_MATCH_PROB):
    rows = []
    for stem, gt, boxes, confs, probs in cache:
        kept = [probs[k] for k in agnostic_keep(boxes, confs, agnostic_iou)]
        result = diagnose_image(kept, no_match_prob=no_match_prob)
        rows.append({
            "stem": stem,
            "gt_verdict": gt.verdict,
            "gt_kind": gt.kind or "normal",
            "gt_chromosomes": ";".join(gt.chromosomes),
            "pred_verdict": result.verdict,
            "pred_chromosomes": ";".join(a.chromosome for a in result.abnormalities),
            "pred_direction": direction(result.abnormalities),
        })
    return score(aggregate_by_case(rows))


def print_row(label, s):
    print(f"  {label:<22} FP={s['fp']:<5} acc={s['accuracy']:.3f}  recall={s['recall']:.3f}", flush=True)


if __name__ == "__main__":
    cache = load_cache()

    print(f"\nAGNOSTIC_IOU sweep (NO_MATCH_PROB={NO_MATCH_PROB}):")
    for iou in [1.0, 0.98, 0.96, 0.95, 0.94, 0.93, 0.92, 0.9, 0.85]:
        print_row(f"agnostic_iou={iou}", evaluate_settings(cache, agnostic_iou=iou))

    print(f"\nNO_MATCH_PROB sweep (AGNOSTIC_IOU={AGNOSTIC_IOU}):")
    for nmp in [1e-3, 1e-4, 5e-5, 3e-5, 2e-5, 1.8e-5, 1.6e-5, 1e-5]:
        print_row(f"no_match_prob={nmp}", evaluate_settings(cache, no_match_prob=nmp))
