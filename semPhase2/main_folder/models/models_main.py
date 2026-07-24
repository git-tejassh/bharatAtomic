import numpy as np
from sklearn.metrics import precision_recall_curve


"""
Canonical, model-agnostic train/val/test split for the EMPS Supervisely export.

Produces ONE deterministic split (hash-based on source-image stem -- stable
even if new images get added to the dataset later) and exports it in THREE
formats, all derived from the same manifest, so LBMS-SAM, YOLO, and RF-DETR
are all evaluated against byte-identical test images:

    1. Generic COCO (source of truth):
        canonical_root/coco/images/<split>/*.png
        canonical_root/coco/annotations/<split>_annotations.json
       -> consumed directly by LBMS-SAM (adjust to your loader's arg names).

    2. YOLO:
        canonical_root/yolo/images/<split>/*.png
        canonical_root/yolo/labels/<split>/*.txt
        canonical_root/yolo/dataset.yaml
       -> consumed by ultralytics YOLO.train()/val().

    3. RF-DETR (Roboflow layout -- verified against rfdetr==1.8.3
       rfdetr/datasets/coco.py:build_roboflow_from_coco, which hardcodes
       this exact structure and file name):
        canonical_root/rfdetr/train/_annotations.coco.json + images
        canonical_root/rfdetr/valid/_annotations.coco.json + images   <- "valid", not "val"
        canonical_root/rfdetr/test/_annotations.coco.json  + images
       -> consumed by RFDETRSegSmall/.train(dataset_dir=...) and
          sv.DetectionDataset.from_coco() for eval.

This replaces:
    - DataLoading.split_images()'s random.shuffle()-based split (order-
      dependent, and split_fraction=0.998 in the notebook leaves a ~0.2%
      test set -- not usable).
    - The RF-DETR notebook's manually-copied "test_dir copy" folder, which
      had no verified relationship to the YOLO split.

Usage:
    python canonical_split.py
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import yolo.yolo_main
# Reuse the already-correct Supervisely decode/polygon/export logic --
# do not reimplement it. The crop+origin alignment step for Supervisely
# bitmaps is non-obvious and this class already gets it right.
from yolo.yolo_main  import DataLoading


# ---------------------------------------------------------------------------
# Manifest (single source of truth for which image goes to which split)
# ---------------------------------------------------------------------------

def _stable_bucket(key: str) -> float:
    """Deterministic float in [0, 1) from a string. Stable across runs,
    machines, and dataset growth: adding new images never reshuffles
    existing train/val/test assignments, unlike index-based random.shuffle()."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def build_manifest(
    supervisely_root: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed_salt: str = "bharatAtomic-emps-v1",
) -> dict:
    """Assign every (image, annotation) pair to exactly one of train/val/test,
    deterministically, at the source-image level. Each EMPS stem here is one
    full micrograph (no cropping happens in DataLoading -- patchify is
    imported but unused), so this is leakage-safe at the image level; it does
    NOT check whether multiple EMPS images share a parent micrograph upstream
    of this export -- verify that against EMPS metadata separately if you
    haven't already.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1.0"

    loader = DataLoading(path=supervisely_root, output_dir="")  # output_dir unused for manifest building
    images, anns = loader.load_images()
    pairs = loader.match_pairs(images, anns)
    if not pairs:
        raise ValueError("No image/annotation pairs matched -- check match_pairs() / dataset path.")

    manifest = {"train": [], "val": [], "test": []}
    for img_name, ann_name in pairs:
        stem = os.path.splitext(img_name)[0]
        b = _stable_bucket(f"{seed_salt}:{stem}")
        if b < test_frac:
            split = "test"
        elif b < test_frac + val_frac:
            split = "val"
        else:
            split = "train"
        manifest[split].append({"image": img_name, "annotation": ann_name})

    for split, items in manifest.items():
        print(f"{split}: {len(items)} images")
    return manifest


def save_manifest(manifest: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest -> {out_path}")


def load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Generic COCO export -- source of truth, feeds LBMS-SAM directly and
#    both adapters below.
# ---------------------------------------------------------------------------

def export_coco(
    manifest: dict,
    supervisely_root: str,
    output_dir: str,
    category_name: str = "particle",
    category_id: int = 1,
) -> None:
    """Export each split from the fixed manifest -- not a fresh shuffle.
    Writes output_dir/images/<split>/ + output_dir/annotations/<split>_annotations.json.
    """
    loader = DataLoading(
        path=supervisely_root,
        output_dir=output_dir,
        category_name=category_name,
        category_id=category_id,
    )
    for split, items in manifest.items():
        pairs = [(item["image"], item["annotation"]) for item in items]
        loader.export_coco_split(split, pairs)


# ---------------------------------------------------------------------------
# 2. YOLO adapter -- explicit converter derived from the generic COCO json
#    above. Avoids ultralytics.data.converter.convert_coco()'s filename-
#    derived output folder naming, which caused the recurring label folder
#    mismatches already logged in project notes.
# ---------------------------------------------------------------------------

def coco_to_yolo_labels(coco_json_path: str, out_labels_dir: str) -> None:
    """One .txt per image, named to match the image stem exactly."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    categories = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}
    images_by_id = {img["id"]: img for img in coco["images"]}

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    out_dir = Path(out_labels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img_info in images_by_id.items():
        h, w = img_info["height"], img_info["width"]
        stem = Path(img_info["file_name"]).stem
        lines = []
        for ann in anns_by_image.get(img_id, []):
            cls = categories[ann["category_id"]]
            for polygon in ann["segmentation"]:
                if len(polygon) < 6:
                    continue  # mirrors the min_points guard in mask_to_coco_polygon
                coords = np.array(polygon, dtype=np.float64).reshape(-1, 2)
                coords[:, 0] /= w
                coords[:, 1] /= h
                coords = np.clip(coords, 0.0, 1.0)
                flat = " ".join(f"{v:.6f}" for v in coords.flatten())
                lines.append(f"{cls} {flat}")
        (out_dir / f"{stem}.txt").write_text("\n".join(lines))
        # An empty-but-present label file for zero-instance images is
        # intentional: ultralytics treats that as "no objects", distinct
        # from a missing label file, which it flags/skips instead.


def export_yolo(coco_root: str, yolo_root: str, splits=("train", "val", "test")) -> None:
    """coco_root is the output_dir passed to export_coco(). Copies images
    across (rather than reusing them in place) so the yolo/ tree is fully
    self-contained and independent of the coco/ tree's layout."""
    for split in splits:
        json_path = Path(coco_root) / "annotations" / f"{split}_annotations.json"
        if not json_path.exists():
            print(f"skip {split}: {json_path} not found")
            continue

        src_img_dir = Path(coco_root) / "images" / split
        dst_img_dir = Path(yolo_root) / "images" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        for img_path in src_img_dir.iterdir():
            shutil.copy2(img_path, dst_img_dir / img_path.name)

        labels_dir = Path(yolo_root) / "labels" / split
        coco_to_yolo_labels(str(json_path), str(labels_dir))
        print(f"{split}: wrote YOLO images -> {dst_img_dir}, labels -> {labels_dir}")


def write_dataset_yaml(yolo_root: str, category_name: str = "particle") -> str:
    import yaml
    yaml_path = Path(yolo_root) / "dataset.yaml"
    cfg = {
        "path": str(Path(yolo_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: category_name},
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Saved dataset.yaml -> {yaml_path}")
    return str(yaml_path)


# ---------------------------------------------------------------------------
# 3. RF-DETR adapter -- Roboflow layout, verified against rfdetr==1.8.3
#    rfdetr/datasets/coco.py:
#        is_valid_coco_dataset()        -> checks <root>/train/_annotations.coco.json
#        build_roboflow_from_coco()     -> PATHS = {
#            "train": (root/"train",  root/"train"/"_annotations.coco.json"),
#            "val":   (root/"valid",  root/"valid"/"_annotations.coco.json"),
#            "test":  (root/"test",   root/"test"/"_annotations.coco.json"),
#        }
#    Note the split folder is "valid", not "val" -- this is the #1 way
#    people silently feed RF-DETR an empty val set.
# ---------------------------------------------------------------------------

_RFDETR_SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


def export_rfdetr(coco_root: str, rfdetr_root: str, splits=("train", "val", "test")) -> None:
    """coco_root is the output_dir passed to export_coco(). Copies each
    split's images alongside a renamed copy of its annotation json into
    RF-DETR's expected train/valid/test layout."""
    for split in splits:
        json_path = Path(coco_root) / "annotations" / f"{split}_annotations.json"
        if not json_path.exists():
            print(f"skip {split}: {json_path} not found")
            continue

        dst_dir = Path(rfdetr_root) / _RFDETR_SPLIT_DIRS[split]
        dst_dir.mkdir(parents=True, exist_ok=True)

        src_img_dir = Path(coco_root) / "images" / split
        for img_path in src_img_dir.iterdir():
            shutil.copy2(img_path, dst_dir / img_path.name)

        shutil.copy2(json_path, dst_dir / "_annotations.coco.json")
        print(f"{split}: wrote RF-DETR split -> {dst_dir} ({_RFDETR_SPLIT_DIRS[split]}/)")

    # Self-check, mirrors rfdetr's own is_valid_coco_dataset() so failures
    # surface here instead of deep inside model.train().
    train_check = Path(rfdetr_root) / "train" / "_annotations.coco.json"
    assert train_check.exists(), f"RF-DETR export invalid -- missing {train_check}"
    print(f"RF-DETR layout validated: {train_check} exists.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SUPERVISELY_ROOT = "/Users/tjsss/Desktop/bharatAtomic/semPhase2/main_folder/emps_dataset/emps-DatasetNinja (2)/ds"
    CANONICAL_ROOT = "/Users/tjsss/Desktop/bharatAtomic/semPhase2/main_folder/canonical_split_v1"

    manifest = build_manifest(SUPERVISELY_ROOT, train_frac=0.70, val_frac=0.15, test_frac=0.15)
    save_manifest(manifest, os.path.join(CANONICAL_ROOT, "manifest.json"))

    coco_root = os.path.join(CANONICAL_ROOT, "coco")
    yolo_root = os.path.join(CANONICAL_ROOT, "yolo")
    rfdetr_root = os.path.join(CANONICAL_ROOT, "rfdetr")

    export_coco(manifest, SUPERVISELY_ROOT, coco_root)
    export_yolo(coco_root, yolo_root)
    write_dataset_yaml(yolo_root)
    export_rfdetr(coco_root, rfdetr_root)

    print("\nAll three models eval on the SAME underlying images now:")
    print(f"  LBMS-SAM: {coco_root}/images/test + {coco_root}/annotations/test_annotations.json")
    print(f"  YOLO:     {yolo_root}/dataset.yaml  (test split)")
    print(f"  RF-DETR:  {rfdetr_root}  (dataset_dir arg -- train/valid/test already correctly named)")