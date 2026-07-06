import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
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

path = '/Users/tjsss/Desktop/bharatAtomic/semPhase2/main_folder/emps_dataset/emps-DatasetNinja (2)/ds/' 

class DataLoading():
    def __init__(self, path: str,
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
        """Pair each image with its annotation via startswith, so ordering
        from os.listdir can never misalign img/ann."""
        pairs = []
        used_ann = set()
 
        for img_name in images:
            img_stem = os.path.splitext(img_name)[0]
            
            match = next((a for a in ann if a not in used_ann and a.startswith(img_stem)), None)
            if match is not None:
                pairs.append((img_name, match))
                used_ann.add(match)
            else:
                print(f"Warning: no annotation found for image {img_name}, skipping.")
        print(img_stem)
        return pairs
 
    def split_images(self, images, ann):
        if not self.split_true:
            return
 
        pairs = self.match_pairs(images, ann)
 
        random.seed(self.seed)
        random.shuffle(pairs)
 
        split_slice = int(self.split_fraction * len(pairs))
        train_val_pairs = pairs[:split_slice]
        test_pairs = pairs[split_slice:]
 
        val_slice = int(self.val_fraction * len(train_val_pairs))
        val_pairs = train_val_pairs[:val_slice]
        train_pairs = train_val_pairs[val_slice:]
 
        splits = {'train_dir': train_pairs, 'val_dir': val_pairs, 'test_dir': test_pairs}
 
        for split_name, split_pairs in splits.items():
            self.export_coco_split(split_name, split_pairs)
            print(f"Successfully converted and exported {len(split_pairs)} items to COCO_format/{split_name}.")
 
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
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
 
    def export_coco_split(self, split_name: str, pairs: list) -> None:
        """
        Convert one split's (image, Supervisely-json) pairs into a single COCO-format
        annotation file, and copy the corresponding images alongside it:
 
            ds/COCO_format/<split_name>/img/<image files>
            ds/COCO_format/<split_name>/annotations.json
 
        Replaces the old raw per-image json copy: COCO consolidates every
        image's instances into one json per split rather than one file per image.
        """
        src_img_dir = os.path.join(self.path, 'img')
        src_ann_dir = os.path.join(self.path, 'ann')
 
        dest_root = os.path.join(self.path, 'COCO_format', split_name)
        dest_img_dir = os.path.join(dest_root, 'img')
        os.makedirs(dest_img_dir, exist_ok=True)
 
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
 
            shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(dest_img_dir, img_name))
 
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
 
                ys, xs = np.where(mask)
                bbox = [
                    int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min()), int(ys.max() - ys.min()),
                ]
 
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": self.category_id,
                    "segmentation": self.mask_to_coco_rle(mask),
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
 
        with open(os.path.join(dest_root, 'annotations.json'), 'w') as f:
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
   
        

    

        
