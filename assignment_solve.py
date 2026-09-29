"""
Classification and assignment: decide what each detected chromosome is by
solving the whole image at once.

Taking each crop's top classifier guess independently fails at the image
level: ~94% per-chromosome accuracy compounds to only ~7% of images fully
right across 46 chromosomes. Instead, every crop's full 24-way probability
vector goes into one global assignment problem (Hungarian algorithm, cost =
-log probability) that fills the 46 expected karyotype slots, so an
ambiguous crop is routed to whichever slot is actually open. Crops left over
after every matching slot is filled are candidate extra copies; slots left
empty are candidate missing ones.
"""

from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from config import NO_MATCH_PROB
from extract_crops import crop_box
from karyotype import EXPECTED_TOTAL, GROUPS, HIGH_CONF_THRESHOLD, Abnormality, KaryotypeResult
from voc_to_yolo import AUTOSOMES, CLASSES, CLASS_TO_ID

_AUTOSOME_SLOTS = [(name, copy) for name in AUTOSOMES for copy in (0, 1)]

# A normal karyotype is XX or XY; the whole problem is solved under each
# template and the cheaper solution wins, so the sex complement is decided
# by the same optimization rather than guessed afterwards.
SLOT_LABELS_XX = _AUTOSOME_SLOTS + [("X", 0), ("X", 1)]
SLOT_LABELS_XY = _AUTOSOME_SLOTS + [("X", 0), ("Y", 0)]
assert len(SLOT_LABELS_XX) == len(SLOT_LABELS_XY) == EXPECTED_TOTAL


def get_crop_probs(classifier, image_path, detections, device="0"):
    """Full 24-class probability vector for each detection's crop."""
    image = Image.open(image_path).convert("RGB")
    probs = []
    for det in detections:
        result = classifier.predict(crop_box(image, *det["bbox"]), verbose=False, device=device)[0]
        vec = np.zeros(len(CLASSES))
        data = result.probs.data.cpu().numpy()
        for idx, name in result.names.items():
            vec[CLASS_TO_ID[name]] = data[idx]
        probs.append(vec)
    return probs


def build_cost_matrix(crop_probs, slot_labels, no_match_prob=NO_MATCH_PROB):
    n_crops, n_slots = len(crop_probs), len(slot_labels)
    # Pad with n_slots dummy rows AND n_crops dummy columns (not just up to
    # the larger side): a square assignment always matches everything, so
    # without both paddings a slot could never stay empty or a crop never be
    # left over, whichever count happens to be larger.
    size = n_crops + n_slots
    cost = np.full((size, size), -np.log(no_match_prob))
    for i, vec in enumerate(crop_probs):
        for j, (label, _) in enumerate(slot_labels):
            cost[i, j] = -np.log(max(vec[CLASS_TO_ID[label]], 1e-6))
    return cost, n_crops, n_slots


def _solve(crop_probs, slot_labels, no_match_prob):
    """Solve one slot template. Returns (total_cost, class_counts,
    excess_confidences, crop_labels); crop_labels[i] is (class, is_extra)."""
    cost, n_crops, n_slots = build_cost_matrix(crop_probs, slot_labels, no_match_prob)
    row_idx, col_idx = linear_sum_assignment(cost)
    total_cost = cost[row_idx, col_idx].sum()

    class_counts = Counter({name: 0 for name in CLASSES})
    crop_labels = [None] * n_crops
    left_over = []
    for r, c in zip(row_idx, col_idx):
        if r >= n_crops:
            continue
        if c < n_slots:
            class_counts[slot_labels[c][0]] += 1
            crop_labels[r] = (slot_labels[c][0], False)
        else:
            left_over.append(r)

    # Each chromosome has only 2 slots, so any count above 2 comes from a
    # left-over crop, counted as its own best guess. Its confidence is kept
    # so a confident extra copy can be told apart from classifier noise.
    excess_confidences = defaultdict(list)
    for r in left_over:
        best = CLASSES[int(np.argmax(crop_probs[r]))]
        class_counts[best] += 1
        excess_confidences[best].append(float(np.max(crop_probs[r])))
        crop_labels[r] = (best, True)

    return total_cost, class_counts, excess_confidences, crop_labels


def _best_template(crop_probs, no_match_prob):
    xx = _solve(crop_probs, SLOT_LABELS_XX, no_match_prob)
    xy = _solve(crop_probs, SLOT_LABELS_XY, no_match_prob)
    return (xx if xx[0] <= xy[0] else xy)[1:]


def assign_crops(crop_probs, no_match_prob=NO_MATCH_PROB):
    """Per-crop (class, is_extra) from the same solution diagnose_image uses."""
    return _best_template(crop_probs, no_match_prob)[2]


def diagnose_image(crop_probs, no_match_prob=NO_MATCH_PROB, high_conf_threshold=HIGH_CONF_THRESHOLD):
    """Assign one image's crops to karyotype slots and flag count mismatches."""
    class_counts, excess_confidences, _ = _best_template(crop_probs, no_match_prob)

    abnormalities = []
    for members in GROUPS.values():
        if sum(class_counts[m] for m in members) == 2 * len(members):
            # The group's total is right, so uneven counts inside it are most
            # likely look-alike members swapped - unless the extra copy comes
            # from a confident left-over crop.
            for name in members:
                actual = class_counts[name]
                if actual > 2 and any(c >= high_conf_threshold for c in excess_confidences.get(name, [])):
                    abnormalities.append(Abnormality(name, expected=2, actual=actual))
            continue
        for name in members:
            if class_counts[name] != 2:
                abnormalities.append(Abnormality(name, expected=2, actual=class_counts[name]))

    x, y = class_counts["X"], class_counts["Y"]
    if (x, y) not in ((2, 0), (1, 1)):
        dist_xx = abs(x - 2) + abs(y - 0)
        dist_xy = abs(x - 1) + abs(y - 1)
        exp_x, exp_y = (1, 1) if dist_xy <= dist_xx else (2, 0)
        if x != exp_x:
            abnormalities.append(Abnormality("X", expected=exp_x, actual=x))
        if y != exp_y:
            abnormalities.append(Abnormality("Y", expected=exp_y, actual=y))

    return KaryotypeResult(total=len(crop_probs), class_counts=class_counts, abnormalities=abnormalities)
