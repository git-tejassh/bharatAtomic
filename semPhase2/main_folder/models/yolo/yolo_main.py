import json
from collections import defaultdict
from pathlib import Path
import random
import os
import numpy as np
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from typing import Optional, Callable
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
from pathlib import Path 
import shutil
import json
from pathlib import Path
from pycocotools import mask as mask_utils
import base64
import cv2
import zlib



from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.utils import TQDM

class COCODataset(YOLODataset):
    """Dataset that reads COCO JSON annotations directly without conversion to .txt files."""

    def __init__(self, *args, json_file="", **kwargs):
        """Initialize the dataset with a COCO JSON annotation file."""
        self.json_file = json_file
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        """Image paths are resolved from the JSON file, not from scanning a directory."""
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        """Parse COCO JSON and convert annotations to YOLO format. Results are saved to a .cache file."""
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        # Sort categories by ID and map to 0-indexed classes
        categories = {cat["id"]: i for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in TQDM(coco["images"], desc="reading annotations"):
            h, w = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["file_name"]
            if not im_file.exists():
                continue

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                if ann.get("iscrowd", False):
                    continue
                # COCO: [x, y, w, h] top-left in pixels -> YOLO: [cx, cy, w, h] center normalized
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2  # top-left to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            lb = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        x["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Load labels from .cache file if available, otherwise parse JSON and create the cache."""
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path)])
            self.im_files = [lb["im_file"] for lb in cache["labels"]]
        except (FileNotFoundError, AssertionError, AttributeError, KeyError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)
        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]
    

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr

class COCOTrainer(DetectionTrainer):
    """Trainer that uses COCODataset for direct COCO JSON training."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build a COCODataset for the given split using the JSON file from the data config."""
        json_file = self.data["train_json"] if mode == "train" else self.data.get("val_json", self.data["train_json"])
        return COCODataset(
            img_path=img_path,
            json_file=json_file,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.model.stride.max()) if hasattr(self, "model") and self.model else 32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
    


import json
from collections import defaultdict
import os
import random
import shutil
import base64
import zlib

import numpy as np
import cv2
from pycocotools import mask as mask_utils


class DataLoading:
    """Converts a Supervisely-exported (img/, ann/) dataset into COCO-format
    train/val/test splits with polygon segmentation, ready for
    ultralytics.data.converter.convert_coco(use_segments=True, ...).

    Output layout under `output_dir`:
        output_dir/
            images/
                train/  <image files>
                val/    <image files>
                test/   <image files>
            annotations/
                train_annotations.json
                val_annotations.json
                test_annotations.json
    """

    def __init__(
        self,
        path: str,
        output_dir: str,
        split_true: bool = True,
        split_fraction: float = 0.70,
        val_fraction: float = 0.20,
        seed: int = 42,
        category_name: str = 'particle',
        category_id: int = 1,
        material_class: str = 'unknown',
    ):
        assert split_fraction < 1, "split fraction must be less than One(1)"
        self.path = path
        self.output_dir = output_dir
        self.split_true = split_true
        self.split_fraction = split_fraction
        self.val_fraction = val_fraction
        self.seed = seed
        # COCO export metadata. material_class isn't present in the
        # Supervisely export at all -- it has to be supplied by the caller,
        # since it's an NFFA-taxonomy label, not something in this dataset's json.
        self.category_name = category_name
        self.category_id = category_id
        self.material_class = material_class

    def match_pairs(self, images, ann):
        """Pair each image with its annotation.

        Primary strategy: exact match on the Supervisely convention, where
        the annotation for 'x.png' is named 'x.png.json'. This is an exact
        equality check via a set, not a prefix check -- avoids the collision
        bug where e.g. 'img_1' would incorrectly match 'img_10.json' under
        startswith(), and is O(1) per lookup instead of O(n) per image.

        Fallback: exact-stem match (still not prefix) for datasets that
        don't follow the "<image>.json" naming convention.
        """
        pairs = []
        ann_set = set(ann)
        used_ann = set()

        for img_name in images:
            expected = f"{img_name}.json"
            if expected in ann_set and expected not in used_ann:
                pairs.append((img_name, expected))
                used_ann.add(expected)
                continue

            img_stem = os.path.splitext(img_name)[0]
            match = next(
                (
                    a for a in ann
                    if a not in used_ann
                    and os.path.splitext(os.path.splitext(a)[0])[0] == img_stem
                ),
                None,
            )
            if match is not None:
                pairs.append((img_name, match))
                used_ann.add(match)
            else:
                print(f"Warning: no annotation found for image {img_name}, skipping.")

        return pairs

    def split_images(self, images, ann):
        if not self.split_true:
            return None

        pairs = self.match_pairs(images, ann)
        if not pairs:
            raise ValueError(
                "No image/annotation pairs matched -- check naming convention "
                "in match_pairs before proceeding."
            )

        random.seed(self.seed)
        random.shuffle(pairs)

        split_slice = int(self.split_fraction * len(pairs))
        train_val_pairs = pairs[:split_slice]
        test_pairs = pairs[split_slice:]

        val_slice = int(self.val_fraction * len(train_val_pairs))
        val_pairs = train_val_pairs[:val_slice]
        train_pairs = train_val_pairs[val_slice:]

        splits = {'train': train_pairs, 'val': val_pairs, 'test': test_pairs}

        for split_name, split_pairs in splits.items():
            self.export_coco_split(split_name, split_pairs)
            print(
                f"Successfully exported {len(split_pairs)} items to "
                f"{self.output_dir}/images/{split_name}/ + "
                f"annotations/{split_name}_annotations.json"
            )

        return splits

    # ---------------- Supervisely -> COCO conversion ----------------

    @staticmethod
    def decode_supervisely_bitmap(bitmap_data: str, origin: list, canvas_size: tuple) -> np.ndarray:
        """
        Decode a single Supervisely 'bitmap' geometry object into a full-resolution
        boolean mask aligned to the source image.

        The Supervisely bitmap payload is zlib-compressed, alpha-channel PNG bytes,
        CROPPED to the object's bounding box -- `origin` is where that crop sits
        in the full image, from `size.height/width` in the annotation json.
        Reshaping the raw bytes directly to the full image size is wrong and will
        silently misalign the mask.

        bitmap_data : object["bitmap"]["data"]    (base64 string)
        origin      : object["bitmap"]["origin"]  ([x, y] top-left of the crop)
        canvas_size : (height, width) of the full image
        """
        compressed = base64.b64decode(bitmap_data)
        decompressed = zlib.decompress(compressed)
        png_bytes = np.frombuffer(decompressed, dtype=np.uint8)
        decoded = cv2.imdecode(png_bytes, cv2.IMREAD_UNCHANGED)

        if decoded is None:
            raise ValueError("Failed to decode Supervisely bitmap PNG payload")

        if decoded.ndim == 3 and decoded.shape[2] >= 4:
            crop_mask = decoded[:, :, 3] > 0
        elif decoded.ndim == 2:
            crop_mask = decoded > 0
        else:
            raise ValueError(f"Unexpected bitmap channel layout: {decoded.shape}")

        full_h, full_w = canvas_size
        x0, y0 = origin
        h, w = crop_mask.shape

        if y0 + h > full_h or x0 + w > full_w:
            raise ValueError(
                f"Bitmap crop [{x0}:{x0+w}, {y0}:{y0+h}] exceeds canvas ({full_h}, {full_w})"
            )

        full_mask = np.zeros((full_h, full_w), dtype=bool)
        full_mask[y0:y0 + h, x0:x0 + w] = crop_mask
        return full_mask

    @staticmethod
    def mask_to_coco_rle(mask: np.ndarray) -> dict:
        """Kept as a utility (e.g. for pycocotools-based mask-IoU eval later).
        NOT used for the `segmentation` field written to the exported JSON --
        ultralytics.data.converter.convert_coco(use_segments=True) expects
        polygons, not RLE; feeding it RLE is what caused the earlier
        merge_multi_segment reshape crash."""
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    @staticmethod
    def mask_to_coco_polygon(mask: np.ndarray, min_points: int = 3) -> list:
        """Extract polygon contours from a binary mask for COCO segmentation.

        Returns a list of flat [x1, y1, x2, y2, ...] lists, one per external
        contour. Contours with too few points, or an odd coordinate count
        (which can't reshape to (N, 2)), are dropped -- this is the exact
        condition that previously crashed Ultralytics' merge_multi_segment.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = []
        for contour in contours:
            flat = contour.flatten().tolist()
            if len(flat) < min_points * 2 or len(flat) % 2 != 0:
                continue
            polygons.append(flat)
        return polygons

    def export_coco_split(self, split_name: str, pairs: list) -> None:
        """
        Convert one split's (image, Supervisely-json) pairs into a single COCO
        annotations file with polygon segmentation, and copy images alongside:

            output_dir/images/<split_name>/<image files>
            output_dir/annotations/<split_name>_annotations.json
        """
        src_img_dir = os.path.join(self.path, 'img')
        src_ann_dir = os.path.join(self.path, 'ann')

        dest_img_dir = os.path.join(self.output_dir, 'images', split_name)
        dest_ann_dir = os.path.join(self.output_dir, 'annotations')
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_ann_dir, exist_ok=True)

        images, annotations = [], []
        ann_id = 0

        for image_id, (img_name, ann_name) in enumerate(pairs):
            ann_path = os.path.join(src_ann_dir, ann_name)
            with open(ann_path) as f:
                data = json.load(f)

            h, w = data["size"]["height"], data["size"]["width"]

            images.append({
                "id": image_id,
                "file_name": img_name,
                "height": h,
                "width": w,
                "material_class": self.material_class,
            })

            shutil.copy(
                os.path.join(src_img_dir, img_name),
                os.path.join(dest_img_dir, img_name),
            )

            for obj in data.get("objects", []):
                if obj.get("geometryType") != "bitmap":
                    print(f"Warning: skipping non-bitmap object in {ann_name} ({obj.get('geometryType')})")
                    continue

                try:
                    mask = self.decode_supervisely_bitmap(
                        obj["bitmap"]["data"], obj["bitmap"]["origin"], (h, w)
                    )
                except (ValueError, KeyError) as e:
                    print(f"Warning: skipping malformed object {obj.get('id')} in {ann_name}: {e}")
                    continue

                if not mask.any():
                    continue

                polygons = self.mask_to_coco_polygon(mask)
                if not polygons:
                    print(
                        f"Warning: skipping object {obj.get('id')} in {ann_name} -- "
                        f"no valid polygon extracted from mask (likely a 1px/degenerate region)."
                    )
                    continue

                ys, xs = np.where(mask)
                bbox_w = int(xs.max() - xs.min())
                bbox_h = int(ys.max() - ys.min())
                if bbox_w <= 0 or bbox_h <= 0:
                    print(
                        f"Warning: skipping object {obj.get('id')} in {ann_name} -- "
                        f"degenerate bbox ({bbox_w}x{bbox_h})."
                    )
                    continue

                bbox = [int(xs.min()), int(ys.min()), bbox_w, bbox_h]

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": self.category_id,
                    "segmentation": polygons,
                    "bbox": bbox,
                    "area": int(mask.sum()),
                    "iscrowd": 0,
                })
                ann_id += 1

        if not annotations:
            print(f"Warning: {split_name} produced zero instances -- check pairing/decode logic.")

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": self.category_id, "name": self.category_name}],
        }

        ann_out_path = os.path.join(dest_ann_dir, f'{split_name}_annotations.json')
        with open(ann_out_path, 'w') as f:
            json.dump(coco, f)

    def load_images(self):
        images_dir = os.path.join(self.path, 'img')
        ann_dir = os.path.join(self.path, 'ann')

        images = sorted(os.listdir(images_dir))
        ann = sorted(os.listdir(ann_dir))

        print(f"Number of images: {len(images)}")
        print(f"Number of annotations: {len(ann)}")

        return images, ann

    def forward(self):
        imgs, anns = self.load_images()
        splits = self.split_images(imgs, anns)
        return splits