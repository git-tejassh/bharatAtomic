import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
from pathlib import Path 
import shutil

path = '/Users/tjsss/Desktop/bharatAtomic/semPhase2/main_folder/emps_dataset/emps-DatasetNinja (2)/ds/' 

class DataLoading():
    def __init__(self, path: str,
                 split_true: bool = True,
                 split_fraction: float = 0.70,
                 val_fraction: float = 0.20,
                 seed: int = 42
    ):
        assert split_fraction < 1, "split fraction must be less than One(1)"
        self.path = path
        self.split_true = split_true
        self.split_fraction = split_fraction
        self.val_fraction = val_fraction
        self.seed = seed

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

        src_img_dir = os.path.join(self.path, 'img')
        src_ann_dir = os.path.join(self.path, 'ann')

        splits = {'train_dir': train_pairs, 'val_dir': val_pairs, 'test_dir': test_pairs}

        for split_name, split_pairs in splits.items():
            dest_img_dir = os.path.join(self.path, split_name, 'img')
            dest_ann_dir = os.path.join(self.path, split_name, 'ann')
            os.makedirs(dest_img_dir, exist_ok=True)
            os.makedirs(dest_ann_dir, exist_ok=True)

            for img_name, ann_name in split_pairs:
                shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(dest_img_dir, img_name))
                shutil.copy(os.path.join(src_ann_dir, ann_name), os.path.join(dest_ann_dir, ann_name))

            print(f"Successfully split and copied {len(split_pairs)} items to {split_name}.")

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
        self.split_images(imgs, anns)
        return imgs, anns

    

    

        
