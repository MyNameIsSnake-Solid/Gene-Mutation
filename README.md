# Chromosome Count Abnormality Detection

Finds every chromosome in a metaphase-spread microscope image, works out which
of the 24 types each one is, and flags **numerical abnormalities**: an extra copy
(trisomy, e.g. Down syndrome) or a missing one (monosomy, e.g. Turner syndrome).
The design goal is to never miss a real abnormality, even at the cost of false alarms.

Dataset: 5,000 annotated images (24 chromosome classes, VOC XML boxes) with
clinical labels in `raw/normal.csv`, `raw/number_abnormalities.csv` and
`raw/structural_abnormalities.csv`, using the dataset's own train/test split.

## Pipeline

1. **Detect** (`detect.py`): a YOLO11s detector boxes every chromosome.
2. **Merge duplicates** (`detect.py`): boxes overlapping more than 95% are merged
   even when the detector gave them different labels. That catches one chromosome
   detected twice, while leaving touching but separate chromosomes alone.
3. **Classify** (`assignment_solve.py`): each box is cropped and a dedicated
   YOLO11n classifier gives a probability for all 24 types.
4. **Assign** (`assignment_solve.py`): instead of trusting each crop's top guess,
   the Hungarian algorithm fits all crops into the 46 expected slots (2 per
   autosome, plus XX or XY; whichever fits better wins) as one global best fit.
   Left-over crops become candidate extra copies, empty slots candidate missing ones.
5. **Flag** (`assignment_solve.py`): counts are compared with 2 per autosome, with a
   tolerance for swaps between look-alike chromosomes in the same group.
6. **Confirm per patient** (`case_aggregate.py`): a finding counts only if another
   image from the same patient shows the same chromosome, and then all of that
   patient's images are flagged. A patient with a single image keeps its own
   verdict, except that a single-image *loss* becomes **needs more cells**
   (losing a chromosome while spreading a cell is a common artifact; the clinical
   standard needs 3 cells to call a loss).

Patients are identified from filenames: every digit but the last is the patient,
and the last digit is the cell (`1053271`–`1053274` are four cells of one patient).
The dataset doesn't document this. It's inferred, but 98% of the resulting 1,524
groups have consistent labels.

## Files

| File | Purpose |
|---|---|
| `config.py` | Model paths and every tuned setting, in one place |
| `evaluate.py` | Runs the full pipeline on the validation set and scores it |
| `viewer.py` | Builds an HTML page showing each image's boxes and verdicts |
| `tune.py` | Sweeps the two tuned settings from cached detections |
| `detect.py`, `assignment_solve.py`, `case_aggregate.py` | The pipeline steps above |
| `karyotype.py`, `scoring.py`, `ground_truth.py` | Shared definitions, scoring, ground-truth loading |
| `voc_to_yolo.py` | Converts the VOC annotations into the YOLO dataset (`raw/chromosome_yolo`) |
| `extract_crops.py`, `extract_crops_from_predictions.py` | Build classifier training data |
| `determinism.py` | Makes GPU inference reproducible; imported first by every entry point |

## Running it

```
python evaluate.py                    # full evaluation, results in runs/evaluate_results.csv
python evaluate.py --no-case-aggregation   # judge each image on its own
python viewer.py                      # example patients -> runs/viewer/karyotype_viewer.html
python viewer.py 105327 10408         # specific patients (filename minus the last digit)
python tune.py                        # delete runs/tune_cache.pkl after retraining a model
```

Always run from the project folder; paths in `config.py` are relative to it.

## Models

- **Detector**: `runs/detect/train-7`. YOLO11s, fine-tuned from an earlier run
  (`train-4`) on `raw/chromosome_yolo` at imgsz 800 for 25 epochs, with 180° rotation
  and up/down and left/right flips. Validation: precision 0.985, recall 0.979, mAP50 0.993.
- **Classifier**: `runs/classify/train-5`. YOLO11n-cls at imgsz 96, 30 epochs, trained on
  crops of the detector's own boxes (see `extract_crops_from_predictions.py`); the
  hand-built balanced subset it used (1,500 per class for training) is kept in
  `raw/chromosome_crops_from_preds_subset`. Top-1 accuracy 93.6%.

## Results (validation set: 1,250 images, 540 patients)

| Stage | False positives | Accuracy | Recall |
|---|---|---|---|
| Detector + counting rules | 600 | 51.7% | 94.7% |
| + classifier and assignment solver | 594 | 52.3% | 100% |
| + duplicate merge | 505 | 59.4% | 100% |
| + patient-level confirmation | 119 | 90.4% | 100% |
| **+ needs more cells (current)** | **75** | **94.0%** | **100%** |

Per patient: 9 of 9 abnormal patients caught, 32 false-alarm patients out of 531
normal ones, and 44 single-image patients marked "needs more cells" (all truly normal).
Accuracy counts "needs more cells" as not flagged.

## Limitations

- **Few positives:** 100% recall rests on 19 abnormal images from 9 patients, so the
  true recall could be much lower.
- **Tuned on the data it's scored on:** both tuned settings were chosen on this
  validation set, and there is no untouched test set. `AGNOSTIC_IOU` sits right at
  an edge: 0.96 or higher misses patient 105101. Lower thresholds keep 100% recall
  and cut false positives further (0.85 gives 60), but that's more tuning on the same
  data. `NO_MATCH_PROB` is robust: anything from 1e-4 to 1e-5 keeps every case.
- **Train/validation overlap:** 127 patients have images in both sets.
- **Out of scope:** structural abnormalities (translocations, deletions) don't change
  the count, so this approach can't see them.
- **Untested path:** there's no single-image monosomy in the validation set, so the
  "needs more cells" handling of a real one is untested.

## Things that didn't work

- **Temperature scaling and stricter group tolerance:** both improved accuracy but
  missed a real case (1051014).
- **Single-class detector:** it had a higher mAP but counted worse. With only one
  class, duplicate removal merged touching chromosomes. It scored 1,072 false
  positives, missed a case at 94.7% recall, and was dropped. Understanding why led
  to the duplicate merge above.

Old experiment data, training runs and retired scripts are in `C:\GeneMutationArchive`,
outside OneDrive.
