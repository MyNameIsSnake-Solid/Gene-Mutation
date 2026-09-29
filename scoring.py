"""
Score per-image verdicts against the clinical ground truth.

Structural abnormalities are reported but excluded from the scored numbers:
a counting approach cannot see them (the count still looks like a normal 46).
"""

from voc_to_yolo import CLASSES

# Ground-truth CSVs name a bare chromosome number ("21"); class names are
# that number prefixed with its group letter ("G21").
NUMBER_TO_CLASS = {name[1:]: name for name in CLASSES if name not in ("X", "Y")}
NUMBER_TO_CLASS["X"] = "X"
NUMBER_TO_CLASS["Y"] = "Y"

# Tokens the dataset uses when it doesn't name a specific chromosome.
AMBIGUOUS_TOKENS = {"UNKNOW", "UNKNOWN", "XX/XY"}


def expected_classes(gt_chromosomes):
    """Map ground-truth tokens to class names. Returns (classes, ambiguous);
    ambiguous means exact-chromosome matching should be skipped."""
    classes = set()
    ambiguous = False
    for token in gt_chromosomes:
        token = token.strip().upper()
        if token in AMBIGUOUS_TOKENS:
            ambiguous = True
            continue
        cls = NUMBER_TO_CLASS.get(token)
        if cls:
            classes.add(cls)
        else:
            ambiguous = True
    return classes, ambiguous


def score(rows):
    gradable = [r for r in rows if r["gt_kind"] != "structural"]
    structural = [r for r in rows if r["gt_kind"] == "structural"]

    tp = fp = tn = fn = 0
    for r in gradable:
        actual_abnormal = r["gt_verdict"] == "abnormal"
        pred_abnormal = r["pred_verdict"] == "abnormal"
        if actual_abnormal and pred_abnormal:
            tp += 1
        elif actual_abnormal:
            fn += 1
        elif pred_abnormal:
            fp += 1
        else:
            tn += 1

    exact_match = partial_match = missed = 0
    numerical_rows = [r for r in gradable if r["gt_kind"] == "numerical"]
    for r in numerical_rows:
        exp_classes, ambiguous = expected_classes(r["gt_chromosomes"].split(";") if r["gt_chromosomes"] else [])
        pred_classes = set(r["pred_chromosomes"].split(";")) if r["pred_chromosomes"] else set()
        if r["pred_verdict"] != "abnormal":
            missed += 1
        elif ambiguous or not exp_classes or (exp_classes & pred_classes):
            exact_match += 1
        else:
            partial_match += 1

    total_gradable = len(gradable)
    return {
        "total_gradable": total_gradable,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / total_gradable if total_gradable else 0,
        "precision": tp / (tp + fp) if (tp + fp) else 0,
        "recall": tp / (tp + fn) if (tp + fn) else 0,
        "numerical_total": len(numerical_rows),
        "exact_match": exact_match, "partial_match": partial_match, "missed": missed,
        "structural_total": len(structural),
        "structural_rows": structural,
    }


def format_report(summary):
    lines = ["=" * 60, "VALIDATION REPORT", "=" * 60]

    lines.append(f"\nGradable images (normal + numerical): {summary['total_gradable']}")
    lines.append(f"  TP={summary['tp']}  FP={summary['fp']}  TN={summary['tn']}  FN={summary['fn']}")
    lines.append(f"  accuracy:  {summary['accuracy']:.3f}")
    lines.append(f"  precision: {summary['precision']:.3f}  (of images flagged abnormal, how many really were)")
    lines.append(f"  recall:    {summary['recall']:.3f}  (of truly abnormal images, how many we caught)")

    lines.append(f"\nOf {summary['numerical_total']} numerically abnormal images:")
    lines.append(f"  exact/plausible chromosome match: {summary['exact_match']}")
    lines.append(f"  flagged abnormal, wrong chromosome: {summary['partial_match']}")
    lines.append(f"  missed entirely (called normal): {summary['missed']}")

    lines.append(f"\nStructural abnormalities: {summary['structural_total']} images "
                 f"(excluded from accuracy above - undetectable by a counting "
                 f"approach; shown for reference only)")
    for r in summary["structural_rows"][:5]:
        lines.append(f"  {r['stem']}: ground truth={r['gt_chromosomes']}  "
                     f"our verdict={r['pred_verdict']} ({r['pred_chromosomes'] or '-'})")

    return "\n".join(lines)
