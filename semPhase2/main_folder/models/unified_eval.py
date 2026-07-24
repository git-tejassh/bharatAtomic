"""
Unified evaluation harness -- scores YOLO, RF-DETR, and LBMS-SAM predictions
through ONE identical pycocotools-based evaluator, on the canonical test set
produced by canonical_split.py.

Why this exists:
    - RF-DETR has no standalone .val()/.evaluate() -- its metrics table only
      appears during .train(), computed via a PyTorch-Lightning callback
      (rfdetr/training/callbacks/coco_eval.py: COCOEvalCallback), using
      torchmetrics.detection.MeanAveragePrecision as the backend.
    - YOLO's model.val() computes mAP with Ultralytics' own AP implementation
      (ultralytics/utils/metrics.py), not pycocotools.
    - These are NOT guaranteed to produce identical numbers on the same
      predictions -- different interpolation/matching details. So even if
      RF-DETR did expose a .val(), comparing its number to YOLO's .val()
      number would still not be a clean comparison.

The fix: extract each model's raw detections (boxes/scores/polygons) into a
common COCO-results format, then score ALL of them with the SAME
pycocotools.cocoeval.COCOeval instance logic. That's what this file does.

Usage:
    gt_json = ".../canonical_split_v1/coco/annotations/test_annotations.json"
    test_img_dir = ".../canonical_split_v1/coco/images/test"

    yolo_dt   = get_yolo_predictions(yolo_model, test_img_dir, gt_json)
    rfdetr_dt = get_rfdetr_predictions(rfdetr_model, test_img_dir, gt_json)
    lbms_dt   = get_lbms_sam_predictions(lbms_model, test_img_dir, gt_json)  # fill in, see stub

    for name, dt in [("YOLO", yolo_dt), ("RF-DETR", rfdetr_dt), ("LBMS-SAM", lbms_dt)]:
        print(f"\n=== {name} (bbox) ===")
        run_coco_eval(gt_json, dt, iou_type="bbox")
        print(f"\n=== {name} (segm) ===")
        run_coco_eval(gt_json, dt, iou_type="segm")
        best_thr, p, r, f1 = sweep_f1(gt_json, dt)
        print(f"{name}: best conf={best_thr:.2f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
"""
import sys
sys.path.append('/Users/tjsss/Desktop/bharatAtomic/semPhase2/main_folder')

import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Reuse the already-correct polygon extraction -- do not reimplement contour
# finding / min-point filtering, it already handles the degenerate cases.
import yolo.yolo_main
from yolo.yolo_main import DataLoading


# ---------------------------------------------------------------------------
# 1. Per-model adapters -> common COCO-results format:
#    [{"image_id": int, "category_id": int, "bbox": [x,y,w,h],
#      "score": float, "segmentation": [[x1,y1,x2,y2,...], ...]}, ...]
# ---------------------------------------------------------------------------

def _filename_to_image_id(gt_json_path: str) -> dict:
    """COCOeval matches on image_id, not filename -- build the lookup once
    from the ground-truth json so every adapter stays in sync with it."""
    with open(gt_json_path) as f:
        gt = json.load(f)
    return {img["file_name"]: img["id"] for img in gt["images"]}


# def get_yolo_predictions(model, image_dir: str, gt_json_path: str, category_id: int = 1) -> list[dict]:
#     """model: an ultralytics.YOLO instance with weights already loaded.
#     Uses results[i].masks.xy (already in original-image pixel coordinates)
#     -- no rasterize/re-extract-contour round trip needed for YOLO specifically.
#     """
#     filename_to_id = _filename_to_image_id(gt_json_path)
#     image_paths = sorted(Path(image_dir).glob("*"))

#     results_out = []
#     for img_path in image_paths:
#         image_id = filename_to_id.get(img_path.name)
#         if image_id is None:
#             continue  # image not in this GT split -- skip rather than silently mismatch ids

#         preds = model.predict(str(img_path), verbose=False)[0]
#         if preds.boxes is None or len(preds.boxes) == 0:
#             continue

#         boxes_xyxy = preds.boxes.xyxy.cpu().numpy()
#         scores = preds.boxes.conf.cpu().numpy()
#         polygons_per_instance = preds.masks.xy if preds.masks is not None else [None] * len(boxes_xyxy)

#         for (x1, y1, x2, y2), score, poly in zip(boxes_xyxy, scores, polygons_per_instance):
#             entry = {
#                 "image_id": int(image_id),
#                 "category_id": category_id,
#                 "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
#                 "score": float(score),
#             }
#             if poly is not None and len(poly) >= 3:
#                 entry["segmentation"] = [poly.flatten().tolist()]
#             results_out.append(entry)
#     return results_out


def get_rfdetr_predictions(model, image_dir: str, gt_json_path: str, threshold: float = 0.01, category_id: int = 1) -> list[dict]:
    """model: an rfdetr RFDETRSegSmall/Medium instance with weights loaded
    (call model.optimize_for_inference() first if you already do that
    elsewhere). threshold is set low (0.01) deliberately -- we want the full
    score range so the F1 sweep below can pick the right operating point;
    filtering to 0.5 here would bake in an unvalidated threshold, same
    mistake as the original notebook.
    """
    filename_to_id = _filename_to_image_id(gt_json_path)
    image_paths = sorted(Path(image_dir).glob("*"))

    results_out = []
    for img_path in image_paths:
        image_id = filename_to_id.get(img_path.name)
        if image_id is None:
            continue

        from PIL import Image
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
                polygons = DataLoading.mask_to_coco_polygon(masks[i].astype(np.uint8))
                if polygons:
                    entry["segmentation"] = polygons
            results_out.append(entry)
    return results_out


def get_lbms_sam_predictions(model, image_dir: str, gt_json_path: str, category_id: int = 1) -> list[dict]:
    """STUB -- I don't have your LBMS-SAM predict/inference method signature
    in context (lbms_sam_main.py wasn't shared in this conversation), so I'm
    not going to guess at field names and risk silently scoring the wrong
    thing. Fill in the three marked lines below to match whatever your
    inference call actually returns (boxes, scores, and either masks or
    polygons per instance) -- the rest of this file (run_coco_eval,
    sweep_f1) will work unchanged once this adapter emits the same
    COCO-results format as the two adapters above.
    """
    filename_to_id = _filename_to_image_id(gt_json_path)
    image_paths = sorted(Path(image_dir).glob("*"))

    results_out = []
    for img_path in image_paths:
        image_id = filename_to_id.get(img_path.name)
        if image_id is None:
            continue

        # >>> REPLACE THIS BLOCK with your actual LBMS-SAM inference call <<<
        raise NotImplementedError(
            "get_lbms_sam_predictions: wire this up to your actual LBMS-SAM "
            "predict() call -- see the docstring above. Expected per-instance "
            "output: box (xyxy), score, and mask (bool array) or polygon."
        )
        # boxes_xyxy = ...
        # scores = ...
        # masks_or_polygons = ...
        #
        # for (x1, y1, x2, y2), score, mask in zip(boxes_xyxy, scores, masks_or_polygons):
        #     entry = {
        #         "image_id": int(image_id),
        #         "category_id": category_id,
        #         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        #         "score": float(score),
        #     }
        #     polygons = DataLoading.mask_to_coco_polygon(mask.astype(np.uint8))
        #     if polygons:
        #         entry["segmentation"] = polygons
        #     results_out.append(entry)
    return results_out


# ---------------------------------------------------------------------------
# 2. Shared scorer -- identical for all three models' output.
# ---------------------------------------------------------------------------

_COCOEVAL_STAT_NAMES = [
    "AP_50:95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]


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
    """xyxy IoU between two boxes."""
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
    """Confidence-threshold sweep with greedy IoU-0.5 one-to-one matching,
    same logic RF-DETR's training-time F1 sweep and YOLO's PR curve are
    both approximating internally -- run explicitly here so it's identical
    across models. Returns (best_threshold, precision, recall, f1) at the
    best-F1 operating point.
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

    best = (0.5, 0.0, 0.0, 0.0)  # threshold, precision, recall, f1
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


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    GT_JSON = "/Users/tjsss/Desktop/bharatAtomic/semPhase2/main_folder/canonical_split_v1/coco/annotations/test_annotations.json"
    TEST_IMG_DIR = "/Users/tjsss/Desktop/bharatAtomic/semPhase2/main_folder/canonical_split_v1/coco/images/test"

    from ultralytics import YOLO
    yolo_model = YOLO("/Users/tjsss/Desktop/bharatAtomic/runs/segment/kfold_demo/average_weights/best.pt")
    yolo_dt = get_yolo_predictions(yolo_model, TEST_IMG_DIR, GT_JSON)

    from rfdetr import RFDETRSegSmall
    rfdetr_model = RFDETRSegSmall(pretrain_weights="/Users/tjsss/Desktop/bharatAtomic/semPhase2/main_folder/models/rf-detr/ckpts/checkpoint_best_total.pth")
    rfdetr_model.optimize_for_inference()
    rfdetr_dt = get_rfdetr_predictions(rfdetr_model, TEST_IMG_DIR, GT_JSON)

    
    # lbms_dt = get_lbms_sam_predictions(lbms_model, TEST_IMG_DIR, GT_JSON)  # fill in adapter first

    for name, dt in [("YOLO", yolo_dt), ("RF-DETR", rfdetr_dt)]:
        print(f"\n=== {name} bbox ===")
        run_coco_eval(GT_JSON, dt, iou_type="bbox")
        print(f"\n=== {name} segm ===")
        run_coco_eval(GT_JSON, dt, iou_type="segm")
        thr, p, r, f1 = sweep_f1(GT_JSON, dt)
        print(f"{name}: best_conf={thr:.2f} P={p:.4f} R={r:.4f} F1={f1:.4f}")
    pass