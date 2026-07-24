"""
eval_rfdetr.py -- standalone evaluation for RF-DETR, meant to run inside
RF-DETR's own venv/kernel (bA_rfdetr, Python 3.11) -- separate from YOLO's
and LBMS-SAM's.

Only dependency beyond rfdetr/supervision: pycocotools, numpy, cv2 (already
present in this venv per the existing notebook's `import cv2`).
    pip install pycocotools

The scoring logic in this file (run_coco_eval, sweep_f1, box IoU) is
DUPLICATED verbatim in eval_yolo.py and eval_lbms_sam.py -- intentionally,
not an oversight. Since each model lives in a separate venv/Python version,
importing a shared module isn't reliable across all three, so identical
code is copy-pasted instead. If you ever change the matching logic
(IoU threshold, sweep range, etc.), change it in all three files the same
way or the "same eval for all three models" guarantee breaks silently.

Notebook usage (inside the RF-DETR venv):
    import torch
    torch.set_default_device("cpu")   # as in the existing rfdetr notebook, before model instantiation

    from rfdetr import RFDETRSegSmall
    from eval_rfdetr import evaluate

    model = RFDETRSegSmall(pretrain_weights="path/to/checkpoint_best_total.pth")
    model.optimize_for_inference()

    results = evaluate(
        model,
        image_dir=".../canonical_split_v1/coco/images/test",
        gt_json_path=".../canonical_split_v1/coco/annotations/test_annotations.json",
    )
"""

import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_IMG_EXTS = {".png", ".jpg", ".jpeg"}
_COCOEVAL_STAT_NAMES = [
    "AP_50:95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]


# ---------------------------------------------------------------------------
# Standalone polygon extraction (same logic as DataLoading.mask_to_coco_polygon
# in yolo_main.py -- reimplemented here rather than imported, since yolo_main.py
# pulls in ultralytics/patchify/tifffile which this venv doesn't have installed).
# ---------------------------------------------------------------------------

def mask_to_coco_polygon(mask: np.ndarray, min_points: int = 3) -> list:
    """Extract polygon contours from a binary mask for COCO segmentation."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        flat = contour.flatten().tolist()
        if len(flat) < min_points * 2 or len(flat) % 2 != 0:
            continue
        polygons.append(flat)
    return polygons


# ---------------------------------------------------------------------------
# Model-specific adapter -> common COCO-results format
# ---------------------------------------------------------------------------

def _filename_to_image_id(gt_json_path: str) -> dict:
    with open(gt_json_path) as f:
        gt = json.load(f)
    return {img["file_name"]: img["id"] for img in gt["images"]}


def get_rfdetr_predictions(model, image_dir: str, gt_json_path: str, threshold: float = 0.01, category_id: int = 1) -> list[dict]:
    """model: an rfdetr RFDETRSegSmall/Medium instance with weights loaded.
    threshold is deliberately low (0.01), not the usual 0.5 -- we want the
    full score range so sweep_f1 can find the real best operating point,
    rather than baking in an unvalidated threshold at collection time.
    """
    filename_to_id = _filename_to_image_id(gt_json_path)
    image_paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in _IMG_EXTS)

    from PIL import Image
    model.optimize_for_inference()
    results_out = []
    for img_path in image_paths:
        image_id = filename_to_id.get(img_path.name)
        if image_id is None:
            continue

        image = Image.open(img_path).convert("RGB")
        detections = model.predict(image, threshold=threshold)

        boxes_xyxy = detections.xyxy
        scores = detections.confidence
        masks = getattr(detections, "mask", None)  # (N, H, W) bool array, or None for box-only variants

        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            entry = {
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[i]),
            }
            if masks is not None:
                polygons = mask_to_coco_polygon(masks[i].astype(np.uint8))
                if polygons:
                    entry["segmentation"] = polygons
            results_out.append(entry)
    return results_out


# ---------------------------------------------------------------------------
# Shared scorer -- KEEP IDENTICAL to eval_yolo.py / eval_lbms_sam.py
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

def evaluate(model, image_dir: str, gt_json_path: str, threshold: float = 0.01, category_id: int = 1) -> dict:
    """Runs bbox eval, segm eval, and the F1 sweep, prints a report, and
    returns everything as a dict for programmatic use (e.g. density
    stratification later)."""
    dt_results = get_rfdetr_predictions(model, image_dir, gt_json_path, threshold=threshold, category_id=category_id)

    bbox_stats = run_coco_eval(gt_json_path, dt_results, iou_type="bbox")
    segm_stats = run_coco_eval(gt_json_path, dt_results, iou_type="segm")
    f1_result = sweep_f1(gt_json_path, dt_results)

    _print_report("RF-DETR", bbox_stats, segm_stats, f1_result)

    return {
        "detections": dt_results,
        "bbox": bbox_stats,
        "segm": segm_stats,
        "f1_sweep": {"threshold": f1_result[0], "precision": f1_result[1], "recall": f1_result[2], "f1": f1_result[3]},
    }


if __name__ == "__main__":
    # Example -- adjust paths to your machine.
    # import torch
    # torch.set_default_device("cpu")
    # from rfdetr import RFDETRSegSmall
    # model = RFDETRSegSmall(pretrain_weights="/Users/tjsss/Desktop/bharatAtomic/semPhase2/main_folder/models/rf-detr/ckpts/checkpoint_best_total.pth")
    # model.optimize_for_inference()
    # evaluate(
    #     model,
    #     image_dir=".../canonical_split_v1/coco/images/test",
    #     gt_json_path=".../canonical_split_v1/coco/annotations/test_annotations.json",
    # )
    pass