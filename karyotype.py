"""
Shared karyotype definitions: expected counts, the standard autosome
groups, and the per-image result type.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from voc_to_yolo import AUTOSOMES

EXPECTED_TOTAL = 46

# A left-over crop at least this confident is treated as a likely real extra
# copy even when its group's total looks normal.
HIGH_CONF_THRESHOLD = 0.75

# Autosomes grouped by letter prefix (A1-A3, B4-B5, C6-C12, ...). Members of
# a group look alike, so the classifier mostly confuses chromosomes within one.
GROUPS = defaultdict(list)
for _name in AUTOSOMES:
    GROUPS[_name[0]].append(_name)


@dataclass
class Abnormality:
    chromosome: str
    expected: int
    actual: int


@dataclass
class KaryotypeResult:
    total: int
    class_counts: Counter
    abnormalities: list = field(default_factory=list)

    @property
    def verdict(self):
        return "abnormal" if self.abnormalities else "normal"
