"""
eval_yolo.py -- standalone evaluation for YOLO, meant to run inside YOLO's
own venv/kernel (separate from RF-DETR's and LBMS-SAM's).

Only dependency beyond ultralytics: pycocotools, numpy.
    pip install pycocotools

The scoring logic in this file (run_coco_eval, sweep_f1, box IoU) is
DUPLICATED verbatim in eval_rfdetr.py and eval_lbms_sam.py -- intentionally,
not an oversight. Since each model lives in a separate venv/Python version,
importing a shared module isn't reliable across all three, so identical
code is copy-pasted instead. If you ever change the matching logic
(IoU threshold, sweep range, etc.), change it in all three files the same
way or the "same eval for all three models" guarantee breaks silently.

Notebook usage (inside the YOLO venv):
    from ultralytics import YOLO
    from eval_yolo import evaluate

    model = YOLO("path/to/best.pt")
    results = evaluate(
        model,
        image_dir=".../canonical_split_v1/coco/images/test",
        gt_json_path=".../canonical_split_v1/coco/annotations/test_annotations.json",
    )
"""

import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_IMG_EXTS = {".png", ".jpg", ".jpeg"}
_COCOEVAL_STAT_NAMES = [
    "AP_50:95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]


# ---------------------------------------------------------------------------
# Model-specific adapter -> common COCO-results format
# ---------------------------------------------------------------------------

def _filename_to_image_id(gt_json_path: str) -> dict:
    with open(gt_json_path) as f:
        gt = json.load(f)
    return {img["file_name"]: img["id"] for img in gt["images"]}


def get_yolo_predictions(model, image_dir: str, gt_json_path: str, category_id: int = 1, device=None) -> list[dict]:
    """model: an ultralytics.YOLO instance with weights already loaded.
    Uses results[i].masks.xy (already in original-image pixel coordinates),
    so no mask-rasterize/re-extract-contour round trip is needed here.
    """
    filename_to_id = _filename_to_image_id(gt_json_path)
    image_paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in _IMG_EXTS)

    results_out = []
    for img_path in image_paths:
        image_id = filename_to_id.get(img_path.name)
        if image_id is None:
            continue  # image not in this GT split -- skip rather than silently mismatch ids

        predict_kwargs = {"verbose": False}
        if device is not None:
            predict_kwargs["device"] = device
        preds = model.predict(str(img_path), **predict_kwargs)[0]
        if preds.boxes is None or len(preds.boxes) == 0:
            continue

        boxes_xyxy = preds.boxes.xyxy.cpu().numpy()
        scores = preds.boxes.conf.cpu().numpy()
        polygons_per_instance = preds.masks.xy if preds.masks is not None else [None] * len(boxes_xyxy)

        for (x1, y1, x2, y2), score, poly in zip(boxes_xyxy, scores, polygons_per_instance):
            entry = {
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
            if poly is not None and len(poly) >= 3:
                entry["segmentation"] = [poly.flatten().tolist()]
            results_out.append(entry)
    return results_out


# ---------------------------------------------------------------------------
# Shared scorer -- KEEP IDENTICAL to eval_rfdetr.py / eval_lbms_sam.py
# ---------------------------------------------------------------------------

def run_coco_eval(gt_json_path: str, dt_results: list[dict], iou_type: str = "bbox") -> dict:
    """iou_type: "bbox" or "segm". Returns the standard 12-stat COCO summary
    dict, computed identically regardless of which model produced dt_results.
    """
    if not dt_results:
        print("No detections -- skipping (would raise inside pycocotools' loadRes).")
        return {name: 0.0 for name in _COCOEVAL_STAT_NAMES}

    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(dt_results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # prints the standard 12-line table

    return dict(zip(_COCOEVAL_STAT_NAMES, coco_eval.stats.tolist()))


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def sweep_f1(
    gt_json_path: str,
    dt_results: list[dict],
    iou_thresh: float = 0.5,
    thresholds: np.ndarray = np.linspace(0.05, 0.95, 19),
) -> tuple[float, float, float, float]:
    """Confidence-threshold sweep with greedy IoU-0.5 one-to-one matching.
    Returns (best_threshold, precision, recall, f1) at the best-F1 point.
    """
    coco_gt = COCO(gt_json_path)
    gt_by_image = {}
    for img_id in coco_gt.getImgIds():
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
        gt_by_image[img_id] = np.array(boxes) if boxes else np.zeros((0, 4))

    dt_by_image = {}
    for d in dt_results:
        img_id = d["image_id"]
        x, y, w, h = d["bbox"]
        dt_by_image.setdefault(img_id, []).append((np.array([x, y, x + w, y + h]), d["score"]))

    best = (0.5, 0.0, 0.0, 0.0)
    for t in thresholds:
        tp = fp = fn = 0
        for img_id, gt_boxes in gt_by_image.items():
            dets = [d for d in dt_by_image.get(img_id, []) if d[1] >= t]
            dets.sort(key=lambda d: -d[1])
            matched_gt = set()
            for box, _score in dets:
                best_iou, best_j = 0.0, -1
                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    iou = _box_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thresh:
                    tp += 1
                    matched_gt.add(best_j)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        if f1 > best[3]:
            best = (float(t), precision, recall, f1)

    return best


def _print_report(name: str, bbox_stats: dict, segm_stats: dict, f1_result: tuple) -> None:
    thr, p, r, f1 = f1_result
    print(f"\n{'=' * 60}")
    print(f"{name} -- unified eval (identical scoring across YOLO/RF-DETR/LBMS-SAM)")
    print(f"{'=' * 60}")
    print(f"{'':12}{'mAP@50:95':>12}{'mAP@50':>12}{'mAP@75':>12}{'mAR@100':>12}")
    print(f"{'bbox':12}{bbox_stats['AP_50:95']:>12.4f}{bbox_stats['AP_50']:>12.4f}"
          f"{bbox_stats['AP_75']:>12.4f}{bbox_stats['AR_100']:>12.4f}")
    print(f"{'segm':12}{segm_stats['AP_50:95']:>12.4f}{segm_stats['AP_50']:>12.4f}"
          f"{segm_stats['AP_75']:>12.4f}{segm_stats['AR_100']:>12.4f}")
    print(f"\nBest-F1 operating point (conf >= {thr:.2f}): "
          f"Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")


# ---------------------------------------------------------------------------
# Orchestrator -- the one function you call from the notebook.
# ---------------------------------------------------------------------------

def evaluate(model, image_dir: str, gt_json_path: str, category_id: int = 1, device=None) -> dict:
    """Runs bbox eval, segm eval, and the F1 sweep, prints a report, and
    returns everything as a dict for programmatic use (e.g. density
    stratification later)."""
    dt_results = get_yolo_predictions(model, image_dir, gt_json_path, category_id=category_id, device=device)

    bbox_stats = run_coco_eval(gt_json_path, dt_results, iou_type="bbox")
    segm_stats = run_coco_eval(gt_json_path, dt_results, iou_type="segm")
    f1_result = sweep_f1(gt_json_path, dt_results)

    _print_report("YOLO", bbox_stats, segm_stats, f1_result)

    return {
        "detections": dt_results,
        "bbox": bbox_stats,
        "segm": segm_stats,
        "f1_sweep": {"threshold": f1_result[0], "precision": f1_result[1], "recall": f1_result[2], "f1": f1_result[3]},
    }


if __name__ == "__main__":
    # Example -- adjust paths to your machine.
    # from ultralytics import YOLO
    # model = YOLO("/Users/tjsss/Desktop/bharatAtomic/runs/segment/kfold_demo/yolo_ft_v2_fold_1/weights/best.pt")
    # evaluate(
    #     model,
    #     image_dir=".../canonical_split_v1/coco/images/test",
    #     gt_json_path=".../canonical_split_v1/coco/annotations/test_annotations.json",
    # )
    pass