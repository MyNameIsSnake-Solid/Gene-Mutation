import csv
import os
from dataclasses import dataclass, field

from voc_to_yolo import TRAIN_TXT

RAW_ROOT = os.path.dirname(TRAIN_TXT)
NORMAL_CSV = os.path.join(RAW_ROOT, "normal.csv")
NUMBER_ABN_CSV = os.path.join(RAW_ROOT, "number_abnormalities.csv")
STRUCTURAL_ABN_CSV = os.path.join(RAW_ROOT, "structural_abnormalities.csv")


@dataclass
class GroundTruth:
    verdict: str                       
    kind: str = None                  
    chromosomes: list = field(default_factory=list)  
    reported_total: int = None        
    note: str = ""                     


def _stem(filename):
    return os.path.splitext(filename.strip())[0]


def _parse_chromosome_list(raw):
    raw = raw.strip()
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _load_csv_rows(path):
    if not os.path.isfile(path):
        print(f"  [warn] missing ground-truth file: {path}")
        return
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            filename = row[0]
            total = int(row[1]) if len(row) > 1 and row[1].strip() else None
            note = row[2] if len(row) > 2 else ""
            yield filename, total, note


def load_ground_truth(csv_dir=None):
    normal_csv = NORMAL_CSV
    number_csv = NUMBER_ABN_CSV
    structural_csv = STRUCTURAL_ABN_CSV
    if csv_dir:
        normal_csv = os.path.join(csv_dir, "normal.csv")
        number_csv = os.path.join(csv_dir, "number_abnormalities.csv")
        structural_csv = os.path.join(csv_dir, "structural_abnormalities.csv")

    truth = {}

    for filename, total, note in _load_csv_rows(normal_csv):
        truth[_stem(filename)] = GroundTruth(
            verdict="normal", kind=None,
            chromosomes=_parse_chromosome_list(note),
            reported_total=total, note=note,
        )

    for filename, total, note in _load_csv_rows(number_csv):
        stem = _stem(filename)
        if stem in truth:
            print(f"  [warn] {stem} appears in both normal.csv and "
                  f"number_abnormalities.csv - keeping the abnormal entry")
        truth[stem] = GroundTruth(
            verdict="abnormal", kind="numerical",
            chromosomes=_parse_chromosome_list(note),
            reported_total=total, note=note,
        )

    for filename, total, note in _load_csv_rows(structural_csv):
        stem = _stem(filename)
        if stem in truth:
            print(f"  [warn] {stem} appears in both an earlier CSV and "
                  f"structural_abnormalities.csv - keeping the structural entry")
        truth[stem] = GroundTruth(
            verdict="abnormal", kind="structural",
            chromosomes=_parse_chromosome_list(note),
            reported_total=total, note=note,
        )

    return truth


if __name__ == "__main__":
    from collections import Counter
    import glob
    from voc_to_yolo import XML_ROOT

    truth = load_ground_truth()
    by_kind = Counter(gt.kind or "normal" for gt in truth.values())
    print(f"loaded {len(truth)} ground-truth entries: {dict(by_kind)}")

    checks = [
        ("104231", "normal", None, []),
        ("1091595", "normal", None, ["Y"]),
        ("103081", "abnormal", "numerical", ["unknow"]),
        ("1056995", "abnormal", "structural", ["8", "17"]),
    ]
    passed = 0
    for stem, exp_verdict, exp_kind, exp_chroms in checks:
        gt = truth.get(stem)
        ok = (gt is not None and gt.verdict == exp_verdict
              and gt.kind == exp_kind and gt.chromosomes == exp_chroms)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {stem}: {gt}")
    print(f"{passed}/{len(checks)} spot checks passed")

    xml_stems = {os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(XML_ROOT, "**", "*.xml"), recursive=True)}
    truth_stems = set(truth)
    print(f"\nXML files: {len(xml_stems)}  ground-truth entries: {len(truth_stems)}")
    print(f"in XML but no ground truth: {len(xml_stems - truth_stems)}")
    print(f"in ground truth but no XML: {len(truth_stems - xml_stems)}")
