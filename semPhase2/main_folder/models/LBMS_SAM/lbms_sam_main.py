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
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL.Image import Image
from lbms_mask_loss import LBMS_MaskLoss
import torch.nn.functional as F

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
   
        
        

def sample_point_prompt(mask: np.ndarray) -> tuple[int, int]:
    """Point deepest inside the mask -- more robust than a random foreground pixel."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)


class LBMSCocoDataset(Dataset):
    def __init__(self, coco_json_path: str, image_dir: str, target_size: int = 1024, 
                 max_samples: Optional[int] = None):
        self.coco = COCO(coco_json_path)
        self.image_dir = image_dir
        self.img_ids = list(self.coco.imgs.keys())

        if max_samples is not None:
            self.img_ids = self.img_ids[:max_samples]

        self.target_size = target_size
        

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        if not anns:
            raise ValueError(f"Image {img_id} ({img_info['file_name']}) has zero instances")

        image = cv2.imread(f"{self.image_dir}/{img_info['file_name']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        ann = anns[np.random.randint(len(anns))]
        assert list(ann["segmentation"]["size"]) == [orig_h, orig_w], (
            f"RLE size {ann['segmentation']['size']} != image size {(orig_h, orig_w)} "
            f"for {img_info['file_name']} -- stale annotation or wrong image file"
        )
        gt_mask = mask_utils.decode(ann["segmentation"]).astype(np.float32)

        # Prompt computed at ORIGINAL resolution, then scaled -- not the other way round
        px, py = sample_point_prompt(gt_mask)
        scale = self.target_size / max(orig_h, orig_w)
        px_scaled, py_scaled = px * scale, py * scale

        image_resized = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))
        mask_resized = cv2.resize(gt_mask, (int(orig_w * scale), int(orig_h * scale)),
                                   interpolation=cv2.INTER_NEAREST)

        # Pad up to target_size x target_size (SAM2 expects square input)
        padded_image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        padded_mask = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        h, w = image_resized.shape[:2]
        padded_image[:h, :w] = image_resized
        padded_mask[:h, :w] = mask_resized

        return {
            "image": torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0,
            "point_coords": torch.tensor([[px_scaled, py_scaled]], dtype=torch.float32),
            "point_labels": torch.tensor([1], dtype=torch.int32),
            "gt_mask": torch.from_numpy(padded_mask),
            "material_class": img_info.get("material_class", "unknown"),
        }
    



def build_dataloader(
    coco_root_dir: str,
    target_size: int = 1024,
    max_samples: Optional[int]= None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    coco_root_dir is a split folder like COCO_format/train_dir, expected to contain:
        coco_root_dir/img/<image files>
        coco_root_dir/annotations.json
    """
    ann_file = os.path.join(coco_root_dir, "annotations.json")
    image_dir = os.path.join(coco_root_dir, "img")
 
    dataset = LBMSCocoDataset(
        coco_json_path=ann_file,
        image_dir=image_dir,
        target_size=target_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
 


class TrainingEval:
    def __init__(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        device: str = "cpu",
    ):
        self.model = model
        # If no optimizer is passed, default to Adam over trainable (adapter) params only.
        # If one IS passed, use it as-is -- do not silently override it (this used to
        # discard any optimizer you passed in and hardcode lr=1e-4 regardless).
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
        )
        self.device = device
 
        # Build mask_loss_fn before resolving the default loss_fn, since the
        # default (self.combined_loss) depends on it.
        self.mask_loss_fn = LBMS_MaskLoss(lambda_dice=1.0, lambda_focal=1.0, lambda_bce=1.0)
        self.loss_fn = loss_fn if loss_fn is not None else self.combined_loss
 
        self.model.to(self.device)
 
        self.scheduler = None
        if self.optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-8
            )
 
    def train_one_epoch(self, loader: DataLoader) -> float:
        """
        Runs one training epoch over `loader`. loss_fn(outputs, batch) -> scalar
        tensor, where `outputs` is an LBMSTrainOutput (has .masks/.iou_pred),
        the return type of model.forward_train() -- NOT the inference-only
        forward(), which takes a single prompts dict and returns detached numpy.
        """

        optimizer = self.optimizer
        loss_fn = self.loss_fn
        if optimizer is None:
            raise RuntimeError("train_one_epoch requires an optimizer; none was given to TrainingEval.")

        self.model.train()
        running_loss = 0.0

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            point_coords = batch["point_coords"].to(self.device)
            point_labels = batch["point_labels"].to(self.device)
            gt_masks = batch["gt_mask"].to(self.device)
            material_class = batch["material_class"]
            if torch.is_tensor(material_class):
                material_class = material_class.to(self.device)

            optimizer.zero_grad()
            outputs = self.model.forward_train(
                images=images,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
 
            loss = loss_fn(outputs, {"gt_mask": gt_masks, "material_class": material_class})
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"  batch {step + 1}/{len(loader)} | loss {loss.item():.4f}")
 
        avg_loss = running_loss / max(len(loader), 1)
        print(f"epoch avg loss: {avg_loss:.4f}")
        return avg_loss
    
 
    def compute_iou(self, pred_binary: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        pred_binary, gt_mask: (B, 1, H, W), both {0, 1} float or bool tensors.
        Returns: (B, 1) IoU per sample, NOT reduced to a scalar -- this is a
        per-sample regression target for the IoU head, not a batch-mean metric.
        """
        pred_binary = pred_binary.float()
        gt_mask = gt_mask.float()
 
        intersection = (pred_binary * gt_mask).sum(dim=(2, 3))
        union = (pred_binary + gt_mask - pred_binary * gt_mask).sum(dim=(2, 3))
        return intersection / (union + eps)
 
 
    def combined_loss(self, outputs, batch):
        """
        Default loss_fn. Picks the best of the multimask_output=True mask
        heads per-sample (highest actual IoU vs. gt), matching SAM's own
        training recipe, instead of always training mask head 0.
        """
        gt = batch["gt_mask"]
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)                      # (B,H,W) -> (B,1,H,W)
        elif gt.dim() != 4 or gt.shape[1] != 1:
            raise ValueError(f"expected gt_mask shaped (B,H,W) or (B,1,H,W), got {tuple(gt.shape)}")
 
        pred_masks = outputs.masks           # (B, num_masks, H, W)
        iou_pred = outputs.iou_pred          # (B, num_masks)
        num_masks = pred_masks.shape[1]
 
        with torch.no_grad():
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
            gt_expanded = gt.expand(-1, num_masks, -1, -1)
            ious = self.compute_iou(pred_binary, gt_expanded)     # (B, num_masks)
            best_idx = ious.argmax(dim=1)                          # (B,)
 
        batch_idx = torch.arange(pred_masks.shape[0], device=pred_masks.device)
        best_mask = pred_masks[batch_idx, best_idx].unsqueeze(1)       # (B,1,H,W)
        best_iou_pred = iou_pred[batch_idx, best_idx].unsqueeze(1)     # (B,1)
        best_actual_iou = ious[batch_idx, best_idx].unsqueeze(1)       # (B,1)
 
        mask_loss = self.mask_loss_fn(best_mask, gt)
        iou_loss = F.mse_loss(best_iou_pred, best_actual_iou)
        return mask_loss + iou_loss
 
 
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Runs val/test in eval mode with grad disabled. Returns average loss."""
        self.model.eval()
        running_loss = 0.0
        for batch in loader:
            images = batch["image"].to(self.device)
            point_coords = batch["point_coords"].to(self.device)
            point_labels = batch["point_labels"].to(self.device)
            gt_masks = batch["gt_mask"].to(self.device)
            material_class = batch["material_class"]
            if torch.is_tensor(material_class):
                material_class = material_class.to(self.device)

            outputs = self.model.forward_train(
                image=images,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            loss = self.loss_fn(outputs, {"gt_mask": gt_masks, "material_class": material_class})
            running_loss += loss.item()
 
        return running_loss / max(len(loader), 1)
    
    def save_trainable_state(self, path: str) -> None:
        """
        Saves only params with requires_grad=True (your adapter weights --
        ~1.3M params), not the frozen SAM2 backbone. Assumes you've already
        set requires_grad correctly on self.model before calling train_model.
        """
        trainable_state = {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        if not trainable_state:
            print("Warning: no trainable parameters found -- check requires_grad is set correctly")
        torch.save(trainable_state, path)
    
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 1,
        checkpoint_path: Optional[str] = None,
    ) -> dict:
        """
        Full train+val loop, using self.model / self.optimizer / self.loss_fn /
        self.device (set in __init__). Assumes you've already frozen the SAM2
        backbone and left GSEFE/MDFF/FeatureFusion adapter params trainable --
        this function doesn't touch requires_grad, it just respects whatever's
        already set.
 
        val_loader should be built with deterministic=True (see build_dataloader)
        so val_loss is actually comparable across epochs.
 
        If checkpoint_path is given, saves adapter-only weights whenever val_loss
        improves. Pass val_loader=None to skip validation entirely (e.g. for a
        quick train-only smoke test on a couple of images).
        """
        self.model.to(self.device)
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
 
        for epoch in range(num_epochs):
            print(f"\n============================== Epoch {epoch + 1}/{num_epochs} ==============================")
 
            train_loss = self.train_one_epoch(train_loader)
            history["train_loss"].append(train_loss)
 
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                print(f"epoch {epoch + 1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
 
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
 
                if checkpoint_path is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_trainable_state(checkpoint_path)
                    print(f"  -> new best val_loss, saved adapter weights to {checkpoint_path}")
            else:
                print(f"epoch {epoch + 1}: train_loss={train_loss:.4f}  (no val_loader given)")
 
        return history
    
    
    def run_training(
        self,
        coco_root_dir: str,
        max_samples: Optional[int] = 5,   # <-- the knob: small int for a smoke test, None for full dataset
        num_epochs: int = 1,
        batch_size: int = 1,
        target_size: int = 256,
        checkpoint_path: Optional[str] = None,
    ) -> dict:
        """
        Convenience wrapper: builds a train-only loader from a COCO root and
        runs train_model. Quick-test defaults: max_samples=5, batch_size=1,
        target_size=256, num_epochs=1. For real training, pass max_samples=None
        and your production batch_size/target_size/num_epochs.
 
        NOTE: builds one loader only, no validation split. If you need val_loss
        / checkpointing against a held-out set, build train_loader and val_loader
        yourself (build_dataloader) and call self.train_model(...) directly.
        """
        loader = build_dataloader(
            coco_root_dir=coco_root_dir,
            target_size=target_size,
            max_samples=max_samples,
            batch_size=batch_size,
            shuffle=(max_samples is None),  # keep deterministic ordering while debugging on a subset
        )
 
        print(f"training on {len(loader.dataset)} images")
        return self.train_model(
            train_loader=loader,
            val_loader=None,
            num_epochs=num_epochs,
            checkpoint_path=checkpoint_path,
        )
 

