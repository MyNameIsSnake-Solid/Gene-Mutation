"""
Combine per-image verdicts across images from the same patient/case.

Filenames encode the case: every digit but the last is the case ID, the
last digit is the cell (e.g. 1053271-1053274 are four cells of one
trisomy-21 case). This is inferred, not documented by the dataset, but
98% of the 1,524 resulting groups have consistent ground-truth labels.

A real numerical abnormality should appear in several cells of a case,
while detector counting errors are independent per image. So an image's
abnormal flag is only trusted if another image of the same case flags
the same chromosome; once a case has such a confirmed finding, every
image in it is flagged, so a cell whose own count went wrong is still
reported. Images with no siblings keep their own verdict, except that a
lone image flagging only missing chromosomes gets NEEDS_MORE_CELLS: losing
a chromosome in one cell is a common spreading artifact (the clinical
standard needs 3 cells to call a loss, 2 for a gain), but it isn't called
normal either, so a real single-cell monosomy still gets surfaced.
"""

from collections import Counter, defaultdict

NEEDS_MORE_CELLS = "needs_more_cells"


def case_id(stem):
    return stem[:-1]


def direction(abnormalities):
    """'gain', 'loss', 'both' or '' for one image's flagged abnormalities."""
    gain = any(a.actual > a.expected for a in abnormalities)
    loss = any(a.actual < a.expected for a in abnormalities)
    return "both" if gain and loss else "gain" if gain else "loss" if loss else ""


def _chromosomes(row):
    return {c for c in row["pred_chromosomes"].split(";") if c}


def aggregate_by_case(rows):
    """Return new rows with pred_verdict/pred_chromosomes replaced by the
    case-level decision. Input rows are not modified."""
    by_case = defaultdict(list)
    for r in rows:
        by_case[case_id(r["stem"])].append(r)

    confirmed = {}
    needs_more = set()
    for key, members in by_case.items():
        found = set()
        for r in members:
            if r["pred_verdict"] != "abnormal":
                continue
            others = [s for s in members if s is not r]
            if not others:
                if r.get("pred_direction") == "loss":
                    needs_more.add(key)
                else:
                    found |= _chromosomes(r) or {"?"}
                continue
            for s in others:
                found |= _chromosomes(r) & _chromosomes(s)
        confirmed[key] = found

    out = []
    for r in rows:
        key = case_id(r["stem"])
        r2 = dict(r)
        if key in needs_more:
            r2["pred_verdict"] = NEEDS_MORE_CELLS
        else:
            found = confirmed[key]
            r2["pred_verdict"] = "abnormal" if found else "normal"
            r2["pred_chromosomes"] = ";".join(sorted(found - {"?"}))
        out.append(r2)
    return out


def score_by_case(rows):
    """Patient-level confusion counts: a case is abnormal if any of its
    images is numerically abnormal. Cases containing structural images are
    excluded, matching evaluate.score()."""
    by_case = defaultdict(list)
    for r in rows:
        by_case[case_id(r["stem"])].append(r)

    tp = fp = tn = fn = 0
    missed = []
    needs_more = Counter()
    for key, members in by_case.items():
        kinds = {r["gt_kind"] for r in members}
        if "structural" in kinds:
            continue
        actual = "numerical" in kinds
        if any(r["pred_verdict"] == NEEDS_MORE_CELLS for r in members):
            needs_more["abnormal" if actual else "normal"] += 1
        pred = any(r["pred_verdict"] == "abnormal" for r in members)
        if actual and pred:
            tp += 1
        elif actual:
            fn += 1
            missed.append(key)
        elif pred:
            fp += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    return {
        "cases": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / total if total else 0,
        "recall": tp / (tp + fn) if (tp + fn) else 0,
        "missed": missed,
        "needs_more_cells": needs_more,
    }


def format_case_report(summary):
    nm = summary["needs_more_cells"]
    return (f"\nPer-patient ({summary['cases']} cases): TP={summary['tp']}  FP={summary['fp']}  "
            f"TN={summary['tn']}  FN={summary['fn']}\n"
            f"  accuracy:  {summary['accuracy']:.3f}\n"
            f"  recall:    {summary['recall']:.3f}"
            + (f"  missed cases: {summary['missed']}" if summary["missed"] else "")
            + f"\n  needs more cells: {sum(nm.values())} cases "
              f"(truly normal: {nm['normal']}, truly abnormal: {nm['abnormal']}; "
              f"a truly abnormal one here is counted as a miss above)")
