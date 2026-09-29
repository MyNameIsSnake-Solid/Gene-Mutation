"""Settings for the adopted pipeline, shared by every script that runs it."""

DETECTOR = "runs/detect/train-7/weights/best.pt"
CLASSIFIER = "runs/classify/train-5/weights/best.pt"

# Detection settings (confidence cutoff, per-class NMS IoU, inference size).
CONF = 0.5
IOU = 0.5
IMGSZ = 800

# Boxes overlapping more than this are merged even when the detector gave
# them different labels, catching one chromosome detected twice. 0.96 and
# above misses patient 105101 (images 1051013 and 1051014).
AGNOSTIC_IOU = 0.95

# Assignment-solver probability floor: a crop/slot match less likely than
# this is worse than leaving the slot empty. With the patient-level step,
# every value from 1e-4 to 1e-5 keeps all abnormal cases (71-81 false positives).
NO_MATCH_PROB = 1.8e-5
