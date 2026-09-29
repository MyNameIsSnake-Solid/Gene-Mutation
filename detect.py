"""
Detection step: run the chromosome detector on an image, then merge boxes
that per-class NMS let through as duplicates of one chromosome.
"""

from config import AGNOSTIC_IOU, CONF, IMGSZ, IOU
from voc_to_yolo import CLASSES


def predict_image(model, image_path, conf_threshold=CONF, device="0", iou=IOU, imgsz=IMGSZ):
    """Return [{"class", "confidence", "bbox"}], bbox as (xmin, ymin, xmax, ymax) pixels."""
    results = model.predict(image_path, conf=conf_threshold, iou=iou, imgsz=imgsz,
                            device=device, verbose=False)
    boxes = results[0].boxes
    return [{"class": CLASSES[int(cls_id)], "confidence": float(conf),
             "bbox": tuple(float(v) for v in xyxy)}
            for cls_id, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy)]


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
             + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


def agnostic_keep(boxes, confs, iou_thresh=AGNOSTIC_IOU):
    """Indices of boxes surviving class-agnostic NMS (highest confidence wins).

    Per-class NMS lets one chromosome survive as two boxes when the detector
    gives them different labels; at a high threshold this merges only
    near-identical boxes, leaving touching but separate chromosomes alone."""
    order = sorted(range(len(boxes)), key=lambda k: confs[k], reverse=True)
    kept = []
    for k in order:
        if all(box_iou(boxes[k], boxes[j]) <= iou_thresh for j in kept):
            kept.append(k)
    return kept


def detect(model, image_path, device="0", conf=CONF, iou=IOU, imgsz=IMGSZ, agnostic_iou=AGNOSTIC_IOU):
    """Detector output with duplicate boxes merged - what the pipeline uses."""
    dets = predict_image(model, image_path, conf_threshold=conf, device=device, iou=iou, imgsz=imgsz)
    kept = agnostic_keep([d["bbox"] for d in dets], [d["confidence"] for d in dets], agnostic_iou)
    return [dets[k] for k in kept]
