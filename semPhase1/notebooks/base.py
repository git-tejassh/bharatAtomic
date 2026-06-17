''' 
this file is for calling other model files
this has loading of noisy data
packing of data into custom
load_data func
transforming the images
calc loss function
custom data function
fine tune function
'''

import torch
import torch.nn as nn
import torch.optim as optim
# from models.rdunet import RDUNet 
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
import os
from pytorch_msssim import ssim
from torchmetrics.image import PeakSignalNoiseRatio
from semPhase1.notebooks.training_notebooks.nafnet_imports.NAFNet_arch import NAFNet, NAFBlock, NAFNetLocal, SimpleGate
from semPhase1.notebooks.training_notebooks.nafnet_imports.local_arch import AvgPool2d , Local_Base
import semPhase1.notebooks.training_notebooks.nafnet_imports.arch_util as arch_util
from pathlib import Path
from DISTS_pytorch import DISTS
from torchmetrics.functional.image.dists import deep_image_structure_and_texture_similarity as dists_fn
torch.amp.autocast(device_type="mps")

import time



device = torch.device('mps' if  torch.backends.mps.is_available() else 'cpu')

psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
psnr_metric.to(device)
D = DISTS().to(device)

import os
from PIL import Image
import numpy as np

from pytorch_msssim import ssim


l1 = nn.L1Loss()
mse_fn = nn.MSELoss()



def list_images(parent_path):
    images_list = []
    
    # Traverse the parent directory and its subdirectories
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                try:
                    # Open the image using PIL
                    img = Image.open(img_path).convert('RGB')  # Ensure RGB mode
                    
                    # Convert to a NumPy array (optional, depending on your workflow)
                    img_array = np.array(img)
                    
                    # Append the array or the PIL image object to the list
                    images_list.append(img_array)  # Using NumPy arrays
                    # images_list.append(img)       # Alternatively, using PIL Image objects
                    
                except Exception as e:
                    print(f"Error opening {img_path}: {e}")
    
    return images_list

# Collect all images from subfolders
def collect_images(parent_path):
    all_images = list_images(parent_path)

    # Verify the collected images
    print(f"Total number of images collected: {len(all_images)}")
    if len(all_images) > 0:
        print(f"Shape of the first image: {all_images[0].shape}")
        print(f"Type of the first image: {type(all_images[0])}")

    return all_images


def load_all_data(parent_dir = '/Users/tjsss/Desktop/bharatAtomic/semPhase1/dataset/crop/images' , split = 0.7 , times = 5):
    all_images = collect_images(parent_dir)
    np.random.shuffle(all_images)
    train_arrays = all_images[:int(len(all_images)*split)]
    test_arrays = all_images[int(len(all_images)*split):]
    noise_obj = NoiseImage()
    train_dataset = CustomData(train_arrays, transform= transform_3(256), repeats = times , training = True , noise_obj = noise_obj)
    test_dataset = CustomData(test_arrays , transform = transform_3(256), repeats = 1, training = False, noise_obj = noise_obj)

    train_loader, val_loader, test_loader = load_data(train_dataset, test_dataset, batch_size = 4)
    return train_loader, val_loader, test_loader


def load_pkl(path_train , path_test):
    train_ip , train_op = pkl.load(open(path_train , 'rb'))
    test_ip , test_op = pkl.load(open(path_test , 'rb'))

    training_dataset = list(zip(train_ip, train_op))
    testing_dataset = list(zip(test_ip, test_op))

    return training_dataset , testing_dataset


def augment_data(data, repeats):
    idx = 0
    new_data = []
    for i in range(len(data)):
        for _ in range(repeats):
            new_data.append(data[i])
    return new_data


def transform_3(size):
            return A.Compose([
                A.Resize(size, size), 
            ],
            additional_targets={"output": "image"})


def transform_1(size):
            return A.Compose([
                A.Resize(size, size),
                A.ToGray(1, method = 'weighted_average' , p=1.0), 
            ],
            additional_targets={"output": "image"})


def calc_loss(pred, target,metric = 'ssim', theta=0.2):
    if metric == 'ssim':
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
        ssim_fn = (1.00-ssim_val)
        return ((1 - theta) * l1(pred, target)) + (theta * ssim_fn)
    elif metric == 'dists':
        # DISTS outputs a distance metric (0.0 for a perfect match). 
        # Use the raw score directly as the loss; do NOT subtract from 1.
        dists_loss = D(pred.float(), target.float(), require_grad=True, batch_average=True)
        return ((1 - theta) * l1(pred, target)) + (theta * dists_loss)

        


def all_losses(pred, target, train = True, c=1):
    # rmse_val = mse_fn(pred,target)
    if c == 1:
        pred_rgb = pred.repeat(1,3,1,1)
        target_rgb = target.repeat(1,3,1,1)
        ssim_val = ssim(pred_rgb, target_rgb, data_range=1.0, size_average=True).item()
    else:
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True).item()
    psnr_val = psnr_metric(pred, target).item()
    dists_val = D(pred.float(), target.float(), require_grad=train, batch_average=True).item()
    return ssim_val, psnr_val, dists_val


def dists_loss(pred, target):
    torch.backends.mps.is_available() 
    criterion = D.to(device)
    criterion.eval()
    for param in criterion.parameters():
        param.requires_grad = False
    dists_loss = criterion(pred.float(), target.float(), require_grad=True, batch_average=True).item()
    return dists_loss

class NoiseImage:
    def __init__(self, prob=None):
        self.prob = np.random.uniform(0, 1) if prob is None else prob

    def _clip_like_input(self, out, ref):
        if ref.max() <= 1.0:
            return np.clip(out, 0.0, 1.0)
        return np.clip(out, 0, 255)

    def add_gaussian_noise(self, img, mean=0.0, var=0.01, clip=True, seed=None):
        rng = np.random.default_rng(seed)
        img = img.astype(np.float32)
        sigma = np.sqrt(var)
        noise = rng.normal(mean, sigma, img.shape).astype(np.float32)
        out = img + noise
        return self._clip_like_input(out, img) if clip else out

    def add_salt_pepper_noise(self, img, amount=0.02, s_vs_p=0.5, seed=None):
        rng = np.random.default_rng(seed)
        out = img.copy()
        h, w = img.shape[:2]
        num = int(amount * h * w)

        num_salt = int(num * s_vs_p)
        ys = rng.integers(0, h, num_salt)
        xs = rng.integers(0, w, num_salt)
        salt_val = 1 if out.max() <= 1.0 else 255
        if out.ndim == 2:
            out[ys, xs] = salt_val
        else:
            out[ys, xs, :] = salt_val

        num_pepper = num - num_salt
        ys = rng.integers(0, h, num_pepper)
        xs = rng.integers(0, w, num_pepper)
        if out.ndim == 2:
            out[ys, xs] = 0
        else:
            out[ys, xs, :] = 0

        return out

    def add_speckle_noise(self, img, var=0.04, seed=None):
        rng = np.random.default_rng(seed)
        img = img.astype(np.float32)
        noise = rng.normal(0.0, np.sqrt(var), img.shape).astype(np.float32)
        out = img + img * noise
        return self._clip_like_input(out, img)

    def add_poisson_noise(self, img, scale=255.0, seed=None):
        rng = np.random.default_rng(seed)
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            scaled = np.clip(img, 0, 1) * scale
            noisy = rng.poisson(scaled).astype(np.float32) / scale
            return np.clip(noisy, 0, 1)
        noisy = rng.poisson(np.clip(img, 0, 255)).astype(np.float32)
        return np.clip(noisy, 0, 255)

    # SEM-relevant: mixed counting + electronics noise
    def add_mixed_poisson_gaussian_noise(self, img, poisson_scale=255.0, gauss_var=0.002, seed=None):
        rng = np.random.default_rng(seed)
        out = self.add_poisson_noise(img, scale=poisson_scale, seed=seed).astype(np.float32)
        gauss = rng.normal(0.0, np.sqrt(gauss_var), img.shape).astype(np.float32)
        out = out + gauss
        return self._clip_like_input(out, img)

    # SEM-relevant: horizontal banding / line noise from raster scan
    def add_scanline_noise(self, img, line_var=0.01, seed=None):
        rng = np.random.default_rng(seed)
        img = img.astype(np.float32)
        h, w = img.shape[:2]
        row_noise = rng.normal(0.0, np.sqrt(line_var), (h, 1)).astype(np.float32)

        if img.ndim == 2:
            out = img + row_noise
        else:
            out = img + row_noise[:, :, None]

        return self._clip_like_input(out, img)

    # SEM-relevant: occasional scan-line shift artifact
    def add_line_shift_artifact(self, img, max_shift=4, shift_prob=0.08, seed=None):
        rng = np.random.default_rng(seed)
        out = img.copy()
        h = img.shape[0]

        for y in range(h):
            if rng.random() < shift_prob:
                shift = rng.integers(-max_shift, max_shift + 1)
                out[y] = np.roll(out[y], shift, axis=0 if img.ndim == 2 else 0)

        return out

    # SEM-relevant: slow drift during acquisition
    def add_drift_distortion(self, img, max_drift=6.0, seed=None):
        rng = np.random.default_rng(seed)
        img = img.astype(np.float32)
        h, w = img.shape[:2]
        out = np.empty_like(img)

        cumulative_shift = np.cumsum(rng.normal(0, max_drift / h, size=h))
        cumulative_shift = np.clip(np.round(cumulative_shift).astype(int), -int(max_drift), int(max_drift))

        for y in range(h):
            shift = cumulative_shift[y]
            out[y] = np.roll(img[y], shift, axis=0 if img.ndim == 2 else 0)

        return out

    # SEM-relevant: charging / shading-like low-frequency intensity field
    def add_charging_artifact(self, img, strength=0.25, vertical=True, seed=None):
        rng = np.random.default_rng(seed)
        img = img.astype(np.float32)
        h, w = img.shape[:2]

        if vertical:
            grad = np.linspace(1.0 - strength, 1.0 + strength, h, dtype=np.float32)[:, None]
        else:
            grad = np.linspace(1.0 - strength, 1.0 + strength, w, dtype=np.float32)[None, :]

        # add slight random low-frequency wobble
        wobble = rng.normal(0, strength * 0.08, size=grad.shape).astype(np.float32)
        field = grad + wobble

        if img.ndim == 2:
            out = img * field
        else:
            out = img * field[:, :, None]

        return self._clip_like_input(out, img)

    # rare hot/dead detector pixels, less important than scan artifacts
    def add_detector_streak_pixels(self, img, amount=0.001, seed=None):
        rng = np.random.default_rng(seed)
        out = img.copy()
        h, w = img.shape[:2]
        n = max(1, int(amount * h * w))

        ys = rng.integers(0, h, n)
        xs = rng.integers(0, w, n)

        high = 1 if out.max() <= 1.0 else 255
        vals = rng.choice([0, high], size=n)

        if out.ndim == 2:
            out[ys, xs] = vals
        else:
            out[ys, xs, :] = vals[:, None]

        return out

    def augment_sem(self, img, seed=None):
        rng = np.random.default_rng(seed)
        p = rng.random()

        if p < 0.20:
            return self.add_mixed_poisson_gaussian_noise(
                img,
                poisson_scale=64.0,   # lower scale = stronger shot noise
                gauss_var=0.01,       # was ~0.001-0.002
                seed=seed
            )
        elif p < 0.40:
            return self.add_scanline_noise(
                img,
                line_var=0.40,       # was mild; now clearly visible banding
                seed=seed
            )
        elif p < 0.58:
            return self.add_line_shift_artifact(
                img,
                max_shift=10,          # was 3-4
                shift_prob=0.23,      # was ~0.06-0.08
                seed=seed
            )
        elif p < 0.76:
            return self.add_drift_distortion(
                img,
                max_drift=16.0,       # was ~3-5
                seed=seed
            )
        elif p < 0.90:
            return self.add_charging_artifact(
                img,
                strength=0.60,        # was ~0.18-0.25
                vertical=rng.random() < 0.5,
                seed=seed
            )
        else:
            out = self.add_mixed_poisson_gaussian_noise(
                img,
                poisson_scale=48.0,
                gauss_var=0.008,
                seed=seed
            )
            out = self.add_scanline_noise(
                out,
                line_var=0.012,
                seed=None if seed is None else seed + 1
            )
            out = self.add_drift_distortion(
                out,
                max_drift=8.0,
                seed=None if seed is None else seed + 2
            )
            out = self.add_line_shift_artifact(
                out,
                max_shift=6,
                shift_prob=0.12,
                seed=None if seed is None else seed + 3
            )
            return out

    def new_augment_sem(self, img, seed=None):
        rng = np.random.default_rng(seed)
        out = img.copy()

        applied = {
            "poisson_gaussian": False,
            "scanline": False,
            "line_shift": False,
            "drift_distortion": False,
            "charging_artifact": False,
            "detector_streaks": False,
        }

        if rng.random() < 0.85:
            poisson_scale = rng.uniform(32.0, 128.0)
            gauss_var = rng.uniform(0.002, 0.012)
            out = self.add_mixed_poisson_gaussian_noise(
                out,
                poisson_scale=poisson_scale,
                gauss_var=gauss_var,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["poisson_gaussian"] = {
                "fired": True,
                "poisson_scale": float(poisson_scale),
                "gauss_var": float(gauss_var),
            }

        if rng.random() < 0.55:
            line_var = rng.uniform(0.004, 0.018)
            out = self.add_scanline_noise(
                out,
                line_var=line_var,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["scanline"] = {
                "fired": True,
                "line_var": float(line_var),
            }

        if rng.random() < 0.35:
            max_shift = int(rng.uniform(3, 10))
            shift_prob = rng.uniform(0.05, 0.20)
            out = self.add_line_shift_artifact(
                out,
                max_shift=max_shift,
                shift_prob=shift_prob,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["line_shift"] = {
                "fired": True,
                "max_shift": max_shift,
                "shift_prob": float(shift_prob),
            }

        if rng.random() < 0.40:
            max_drift = rng.uniform(4.0, 14.0)
            out = self.add_drift_distortion(
                out,
                max_drift=max_drift,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["drift_distortion"] = {
                "fired": True,
                "max_drift": float(max_drift),
            }

        if rng.random() < 0.30:
            strength = rng.uniform(0.15, 0.45)
            vertical = rng.random() < 0.5
            out = self.add_charging_artifact(
                out,
                strength=strength,
                vertical=vertical,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["charging_artifact"] = {
                "fired": True,
                "strength": float(strength),
                "vertical": bool(vertical),
            }

        if rng.random() < 0.20:
            amount = rng.uniform(0.0005, 0.002)
            out = self.add_detector_streak_pixels(
                out,
                amount=amount,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["detector_streaks"] = {
                "fired": True,
                "amount": float(amount),
            }

        return {
            "x": out,
            "y": img,
            "noise_added": applied,
        }
    
    def mid_augment_sem(self, img, seed=None):
        rng = np.random.default_rng(seed)
        out = img.copy()

        applied = {
            "poisson_gaussian": False,
            "scanline": False,
            "line_shift": False,
            "drift_distortion": False,
            "charging_artifact": False,
            "detector_streaks": False,
        }

        ## POISSON NOISE
        if rng.random() < 0.85:
            poisson_scale = rng.uniform(16.0, 96.0)
            gauss_var = rng.uniform(0.008, 0.035)
            out = self.add_mixed_poisson_gaussian_noise(
                out,
                poisson_scale=poisson_scale,
                gauss_var=gauss_var,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["poisson_gaussian"] = {
                "fired": True,
                "poisson_scale": float(poisson_scale),
                "gauss_var": float(gauss_var),
            }

        ## SCANLINE NOISE
        if rng.random() < 0.60:
            line_var = rng.uniform(0.02, 0.07)
            out = self.add_scanline_noise(
                out,
                line_var=line_var,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["scanline"] = {
                "fired": True,
                "line_var": float(line_var),
            }

        ## LINE SHIFT NOISE
        if rng.random() < 0.40:
            max_shift = int(rng.uniform(5, 30))
            shift_prob = rng.uniform(0.10, 0.40)
            out = self.add_line_shift_artifact(
                out,
                max_shift=max_shift,
                shift_prob=shift_prob,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["line_shift"] = {
                "fired": True,
                "max_shift": max_shift,
                "shift_prob": float(shift_prob),
            }

        ## DRIFT NOISE
        if rng.random() < 0.48:
            max_drift = rng.uniform(10.0, 50.0)
            out = self.add_drift_distortion(
                out,
                max_drift=max_drift,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["drift_distortion"] = {
                "fired": True,
                "max_drift": float(max_drift),
            }

        ## CHARGING NOISE
        if rng.random() < 0.38:
            strength = rng.uniform(0.30, 0.65)
            vertical = rng.random() < 0.5
            out = self.add_charging_artifact(
                out,
                strength=strength,
                vertical=vertical,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["charging_artifact"] = {
                "fired": True,
                "strength": float(strength),
                "vertical": bool(vertical),
            }

        ## DETECTOR STREAK
        if rng.random() < 0.20:
            amount = rng.uniform(0.003, 0.015)
            out = self.add_detector_streak_pixels(
                out,
                amount=amount,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["detector_streaks"] = {
                "fired": True,
                "amount": float(amount),
            }

        return {
            "x": out,
            "y": img,
            "noise_added": applied,
        }

    def uf_augment_sem(self, img, seed=None):
        rng = np.random.default_rng(seed)
        out = img.copy()

        applied = {
            "poisson_gaussian": False,
            "scanline": False,
            "line_shift": False,
            "drift_distortion": False,
            "charging_artifact": False,
            "detector_streaks": False,
        }

## POISSON NOISE
        if rng.random() < 0.85:
            poisson_scale = rng.uniform(2.0, 64.0)
            gauss_var = rng.uniform(0.02, 0.08)
            out = self.add_mixed_poisson_gaussian_noise(
                out,
                poisson_scale=poisson_scale,
                gauss_var=gauss_var,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["poisson_gaussian"] = {
                "fired": True,
                "poisson_scale": float(poisson_scale),
                "gauss_var": float(gauss_var),
            }

## SCANLINE NOISE
        if rng.random() < 0.65:
            line_var = rng.uniform(0.05, 0.20)
            out = self.add_scanline_noise(
                out,
                line_var=line_var,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["scanline"] = {
                "fired": True,
                "line_var": float(line_var),
            }

## LINE SHIFT NOISE
        if rng.random() < 0.45:
            if rng.random() > 0.38:
                max_shift = 80
            else:
                max_shift = int(rng.uniform(10, 50))
            shift_prob = rng.uniform(0.3, 0.80)
            out = self.add_line_shift_artifact(
                out,
                max_shift=max_shift,
                shift_prob=shift_prob,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["line_shift"] = {
                "fired": True,
                "max_shift": max_shift,
                "shift_prob": float(shift_prob),
            }

## DRIFT NOISE
        if rng.random() < 0.55:
            max_drift = rng.uniform(20.0, 150.0)
            out = self.add_drift_distortion(
                out,
                max_drift=max_drift,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["drift_distortion"] = {
                "fired": True,
                "max_drift": float(max_drift),
            }

## CHARGING NOISE
        if rng.random() < 0.45:
            strength = rng.uniform(0.5, 1.0)
            vertical = rng.random() < 0.5
            out = self.add_charging_artifact(
                out,
                strength=strength,
                vertical=vertical,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["charging_artifact"] = {
                "fired": True,
                "strength": float(strength),
                "vertical": bool(vertical),
            }
## DETECTOR STREAK
        if rng.random() < 0.20:
            amount = rng.uniform(0.01, 0.05)
            out = self.add_detector_streak_pixels(
                out,
                amount=amount,
                seed=int(rng.integers(0, 1_000_000))
            )
            applied["detector_streaks"] = {
                "fired": True,
                "amount": float(amount),
            }

        return {
            "x": out,
            "y": img,
            "noise_added": applied,
        }
    
    def build_augmentation_split(self, dataset, seed=42):
        n = len(dataset)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)

        cut1 = int(n * 0.30)
        cut2 = int(n * 0.60)  # 0.30 + 0.40

        normal_idx = perm[:cut1]
        mid_idx    = perm[cut1:cut2]
        uf_idx     = perm[cut2:]

        assignment = {}
        for i in normal_idx:
            assignment[int(i)] = "normal"
        for i in mid_idx:
            assignment[int(i)] = "mid"
        for i in uf_idx:
            assignment[int(i)] = "uf"

        return assignment
    

    def augment_3way_sem(self, img, idx, assignment, seed=None):
        mode = assignment[idx]
        if mode == "normal":
            return self.new_augment_sem(img, seed=seed)
        elif mode == "mid":
            return self.mid_augment_sem(img, seed=seed)
        else:
            return self.uf_augment_sem(img, seed=seed)



def augment(noisy_images, repeats=5):
    """
    noisy_images: torch.Tensor of shape [N, C, H, W]
    repeats: how many augmented versions per input image
    returns: torch.Tensor of shape [N * repeats, C, H, W]
    """
    assert isinstance(noisy_images, torch.Tensor), "must be torch tensor"
    assert noisy_images.ndim == 4, "input shape must be [N, C, H, W]"

    device = noisy_images.device
    dtype = noisy_images.dtype

    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(0.02, 0.02),
            rotate=(-10, 10),
            shear=(-5, 5),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),
    ])

    imgs = noisy_images.detach().cpu()
    out = []

    for i in range(imgs.shape[0]):
        img = imgs[i]  # [C, H, W]

        for _ in range(repeats):
            x = img.permute(1, 2, 0).numpy()  # [H, W, C]

            if x.shape[2] == 1:
                x_aug = aug(image=x[:, :, 0])["image"]
                x_aug = np.expand_dims(x_aug, axis=-1)
            else:
                x_aug = aug(image=x)["image"]

            x_aug = torch.from_numpy(x_aug).permute(2, 0, 1)  # [C, H, W]
            out.append(x_aug)

    out = torch.stack(out, dim=0).to(device=device, dtype=dtype)
    return out


class CustomData(Dataset):
    def __init__(self, images, transform = None, repeats = 1, training = False, noise_obj = None, aug_assignment = None):
        self.images = images
        self.transform = transform
        self.repeats = repeats
        self.training = training
        self.noise_obj = noise_obj if noise_obj is not None else NoiseImage()
        self.aug_assignment = aug_assignment if aug_assignment is not None else \
            self.noise_obj.build_augmentation_split(self.images, seed=42)
        if transform == transform_1:
            self.transform_type = 'transform_1'  # Grayscale (C=1)
        else:
            self.transform_type = 'transform_3'  # RGB (C=3)



    def __len__(self):
        return len(self.images) * self.repeats if self.training else len(self.images)

    
    def __getitem__(self, idx):
        img_idx = idx % len(self.images) if self.training else idx

        img = self.images[img_idx]
        img = np.asarray(img, dtype=np.float32) / 255.0

        # Start from original clean image
        y_img = img.copy()

        # First: create augmented clean target
        if self.transform is not None:
            aug = self.transform(image=y_img)
            y_img = np.asarray(aug["image"], dtype=np.float32)

        # Then: create noisy input from the augmented target
        if self.noise_obj is not None:
            out = self.noise_obj.augment_3way_sem(y_img.copy(),
                idx=img_idx,
                assignment=self.aug_assignment,
                seed=int(idx))
        else:
            x_img = y_img.copy()
            out = {"x": x_img, "y": y_img, "noise_added": None}

        noise_info = None  # default

        if self.noise_obj:
            x_img = out["x"]
            y_img = out["y"]
            noise_info = out.get("noise_added", None)

        def np_to_tensor(arr):
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                return torch.from_numpy(arr).unsqueeze(0).float()
            elif arr.ndim == 3:
                return torch.from_numpy(arr.transpose(2, 0, 1)).float()
            else:
                raise ValueError(f"Unsupported image ndim: {arr.ndim}")

        x = np_to_tensor(x_img)
        y = np_to_tensor(y_img)

        return x, y
    

def load_data(train_dataset, test_dataset, batch_size, val_ratio=0.2):
    n = len(train_dataset)
    val_size = int(n * val_ratio)
    train_size = n - val_size

    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    return train_loader, val_loader, test_loader


# train_dataset = CustomData(noisy_training_dataset, transform=train_transform_naf, repeats=1, training=True)
# test_dataset = CustomData(noisy_testing_dataset, transform=test_transform_naf, repeats=1, training=False)
# train_loader, val_loader, test_loader = load_data(train_dataset, test_dataset, batch_size=4)


import time

def fineTune(model, train_loader, val_loader, num_epochs=20, theta = 0.4 , name='new_model.pth' , save_freq = 2, metric = 'ssim' , device = 'cpu'):
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-8
)
    
    
    use_amp = (device.type == "mps")
    amp_dtype = torch.bfloat16

    x_sample, y_sample = next(iter(train_loader))
    print(f"x range: [{x_sample.min():.3f}, {x_sample.max():.3f}]")
    print(f"y range: [{y_sample.min():.3f}, {y_sample.max():.3f}]")
    channels = int(x_sample.shape[1])
    # Expected: x range: [-1.0, 1.0], y range: [-1.0, 1.0]
    train_losses, val_losses, epochs_plotted, epochs10 = [], [], [], []
    psnr_scores, psnr_scores_train = [], []
    dists_scores, dists_scores_train = [], []
    epoch_train_losses, epoch_val_losses = [] , []
    dists_score_val, ssim_score_val = [],[]
    ssim_scores = []
    epochs_dists_train, epochs_dists_val , epochs_ssim_train ,epochs_ssim_val = [],[],[],[]
    psnr_train_epochs, dists_train_epochs = [], []
    ssim_val_epochs, dists_val_epochs = [], []



    ## EPOCH
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        psnr_train_epoch= dists_train_epoch = 0.0
        ssim_val_epoch = dists_val_epoch = 0.0

    

    ## TRAINING     
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type="mps", dtype=amp_dtype):
                    pred = model(x).clamp(0,1)
                    loss = calc_loss(pred.float(), y.float(), metric ,theta)
            else:
                pred = model(x).clamp(0,1)
                loss = calc_loss(pred.float(), y.float(), metric ,theta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_losses.append(loss.item())
            
            if (batch_idx + 1) % 10 == 0:
                with torch.no_grad():
                    dists_t = (loss - ((1-theta)*l1(pred, y)))/theta
                    psnr10 = psnr_metric(pred , y).item()

                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.8f} \n PSNR: {psnr10} | DISTS: {dists_t}")
                psnr_scores_train.append(psnr10)
                dists_scores_train.append(dists_t)
        

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.8f} | Time: {epoch_time:.4f}s")
        torch.save(model.state_dict(), name)


        for psnr in psnr_scores_train:
            psnr_train_epoch+= psnr
        psnr_train_epochs.append((psnr_train_epoch)*10/len(train_loader))
        for dists in dists_scores_train:
            dists_train_epoch += dists
        dists_train_epochs.append((dists_train_epoch)*10/(len(train_loader)))

    ## EVALUATION
        model.eval()
        channels = int(x_sample.shape[1])
        with torch.no_grad():
            psnr_sum = val_loss_sum = ssim_score_val_sum = dists_score_val_sum = 0.0
            ssim_score_10 = dists_score_10 = 0.0
            dists_score_val_list = []
            ssim_score_val_list = []
            psnr_metric.reset()

            for batch_idx, (x,y) in enumerate(val_loader):
                x,y = x.to(device) , y.to(device)

                if use_amp:
                    with torch.autocast(device_type="mps", dtype=torch.float16):
                        pred = model(x).clamp(0,1)
                else:
                    pred = model(x).clamp(0, 1)


                pred_f, y_f = pred.float(), y.float()
                val_loss = calc_loss(pred_f, y_f, metric, theta)
                ssim_val, psnr_val, dists_val = all_losses(pred_f, y_f ,train = False ,c = channels)
                val_losses.append(val_loss.item())
                psnr_score = psnr_val
                psnr_sum += psnr_score
                psnr_scores.append(psnr_score) 
                dists_scores.append(dists_val)
                ssim_scores.append(ssim_val)
                val_loss_sum += val_loss.item()
                ssim_score_val_sum += ssim_val
                dists_score_val_sum += dists_val
                dists_score_10 += dists_val
                ssim_score_10 += ssim_val

        
                
                 

                if (batch_idx + 1) % 10 == 0:
                    epochs10.append(epoch+1)
                    print(f"Batch: {batch_idx+1}/{len(val_loader)} | Val Loss: {val_loss.item():.8f}")
                    print(f" DISTS: {(dists_score_10/10):.4f} | SSIM: {(ssim_score_10/10):.4f}")
                    dists_score_val_list.append((dists_score_10/10))
                    ssim_score_val_list.append((ssim_score_10/10))
                    dists_score_10 = 0 
                    ssim_score_10 = 0
            
            avg_val_loss = val_loss_sum / len(val_loader)
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Avg Val loss: {avg_val_loss:.8f} \n Avg PSNR Score: {psnr_sum/len(val_loader):.8f} \n lr: {current_lr} \n Avg DISTS Score: {dists_score_val_list/len(val_loader):.8f} \n Avg SSIM Score: {ssim_score_val_list/len(val_loader):.8f}")

            for ssim in ssim_score_val:
                ssim_val_epoch= ssim
            ssim_val_epochs.append((ssim_score_val_sum)/len(val_loader))
            
        
            for dists in dists_score_val:
                dists_val_epoch+= dists
            dists_val_epochs.append((dists_score_val_sum)/(len(val_loader)))
            
        
        epochs_plotted.append(epoch+1)   
        epochs_dists_val.append(dists_val_epochs)
        


        
        
        


        
        epoch_train_losses.append(total_loss / len(train_loader))
        epoch_val_losses.append(avg_val_loss)

        if (epoch % save_freq) == 0:
            torch.save(model.state_dict(), name)
            

    
    # fineTune(rdunet_model, train_loader, val_loader, num_epochs=1

    # for batch_idx, (x, y) in enumerate(val_loader):
    #     x, y = x.to(device), y.to(device)
    #     pred = model(x)
    #     val_loss = calc_loss(pred, y, theta=0.4)
    #     print(f"  Validation Batch {batch_idx + 1}/{len(val_loader)} | Loss: {val_loss.item():.8f}")

    torch.save(model.state_dict(), name)


##PLOTTING
    plt.plot(epochs_plotted, epoch_train_losses, label="Train Loss", color="blue" , linestyle = "--" , linewidth = 2 )
    plt.plot(epochs_plotted, epoch_val_losses, label="Val Loss", color="red", linewidth = 2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(epochs_plotted, psnr_train_epochs, label="PSNR - Training", color="green", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)") 
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(epochs_plotted, dists_val_epochs, label="DISTS VAL", color="blue" , linestyle = "--" , linewidth = 2 )
    plt.plot(epochs_plotted, dists_train_epochs, label="DISTS TRAIN", color="red", linewidth = 2)
    plt.xlabel("Epoch")
    plt.ylabel("Scores")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(epochs_plotted, ssim_val_epochs, label="SSIM VAL", color="blue" , linestyle = "--" , linewidth = 2 )
    plt.xlabel("Epoch")
    plt.ylabel("Scores")
    plt.legend()
    plt.grid(True)
    plt.show()



def test_func(model, ip_img, transform, augmentation = 1,  channels=3, device='cpu'):
    """
    channels=3 : model expects RGB input  (e.g. AttentionUNet, NAFNet trained on RGB)
    channels=1 : model expects grayscale  (e.g. RDUNet trained on single-channel SEM)
 
    transform   : an Albumentations Compose that at minimum resizes to the target size
                  (e.g. transform_3(256) or transform_1(256))
    """
    assert channels in (1, 3), "channels must be 1 or 3"
 
    device_str = device.type if isinstance(device, torch.device) else str(device)
    model.eval()
 
    # ── 1. Load ────────────────────────────────────────────────────────────────
    with Image.open(ip_img) as img:
        if channels == 3:
            img_array = np.array(img.convert('RGB'))          # (H, W, 3) uint8
        else:
            img_array = np.array(img.convert('L'))            # (H, W)    uint8
 
    # ── 2. Resize FIRST, then noise (matches training pipeline) ───────────────
    if transform is not None:
        aug = transform(image=img_array)
        img_resized = aug["image"]                            # (256,256,3) or (256,256)
    else:
        img_resized = img_array
 
    noise_obj = NoiseImage()
    if augmentation == 1:
        out = noise_obj.new_augment_sem(img_resized) 
    elif augmentation == 2:
        out = noise_obj.mid_augment_sem(img_resized)
    elif augmentation ==3:
        out = noise_obj.uf_augment_sem(img_resized)             # handles 2D and 3D
 
    x_np = out["x"].astype(np.float32) / 255.0               # noisy
    y_np = out["y"].astype(np.float32) / 255.0               # clean
 
    # ── 3. Numpy → Tensor ─────────────────────────────────────────────────────
    if channels == 3:
        # (H, W, 3) → (1, 3, H, W)
        x = torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        y = torch.from_numpy(y_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    else:
        # (H, W) → (1, 1, H, W)
        x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).float().to(device)
        y = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0).float().to(device)
 
    # ── 4. Inference ──────────────────────────────────────────────────────────
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
 
    with torch.inference_mode():
        start = time.time()
        with torch.autocast(device_type=device_str, dtype=torch.float16):
            pred = model(x)
        pred_time = time.time() - start                       # pure model latency
 
        pred  = pred.clamp(0, 1).float()
        y_f   = y.float()
        x_f = x.float()
 
        ssim_score = ssim(pred, y_f, data_range=1.0, size_average=True).item()
        psnr_init = psnr_metric(x_f , y_f)
        psnr_score = psnr_metric(pred, y_f).item()
        _, _, dists_val = all_losses(pred, y_f, train=False, c=channels)
        _,_,dists_init = all_losses(x_f, y_f, train=False, c= channels)
        loss = calc_loss(pred, y_f, theta=0.4)
        pred_cpu = pred.cpu()
 
    total_time = time.time() - start
 
    # ── 5. Metrics ────────────────────────────────────────────────────────────
    print(f"Loss : {loss.item():.8f} | SSIM: {ssim_score:.4f}")
    print(f"PSNR : {psnr_score:.4f} dB | DISTS: {dists_val:.4f}")
    print(f"Intial PSNR : {psnr_init:.4f} , Initial DISTS : {dists_init}")
    print(f"Improvement : {psnr_score - psnr_init} dB")
    print(f"Pred Time: {pred_time:.4f}s  |  Total Time: {total_time:.4f}s")
 
    # ── 6. Visualisation (channel 0 for both 1ch and 3ch) ─────────────────────
    x_vis    = x.cpu()[0, 0].clamp(0, 1).numpy()
    pred_vis = pred_cpu[0, 0].clamp(0, 1).numpy()
    y_vis    = y_f.cpu()[0, 0].clamp(0, 1).numpy()
 
    
    fig, axes = plt.subplots(1, 3, figsize=(25, 25))

    axes[0].imshow(x_vis, cmap='gray')
    axes[0].set_title('Input (Degraded)')
    axes[0].axis('off')

    axes[1].imshow(pred_vis, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    axes[2].imshow(y_vis, cmap='gray')
    axes[2].set_title('Label (Clean)')
    axes[2].axis('off')

    fig.text(
        0.5, 0.02,
        f"Loss: {loss.item():.4f}, SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.2f} dB",
        ha='center', fontsize=10
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def to_1ch(x):
    # x: (B, 3, H, W) or (B, 1, H, W)
    if x.dim() == 4 and x.shape[1] == 3:
        return x.mean(dim=1, keepdim=True)
    return x


def test_func_batches(model, test_loader, device='cpu', transform = None):
    model.eval()
    model.to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    with torch.no_grad():
        test_loss_sum = 0.0
        psnr_score = 0.0
        ssim_val, dists_val = 0.0,0.0

        for batch_id, (x, y) in enumerate(test_loader):        

            x, y = x.to(device), y.to(device)
            # if y.shape[1] == 3:
            #     y = y[:, :1, :, :]

            with torch.autocast('mps' , dtype = torch.float16):
                pred = model(x).to(device)
                test_loss = calc_loss(pred, y, theta=0.4)
            # pred = to_1ch(pred).to(device)

            # if pred.dim() == 4 and pred.shape[1] == 3:
            #     pred = pred[:, :1, :, :]

            pred = pred.clamp(0.0, 1.0).float()

            
            ssim_val, psnr_val, dists_val = all_losses(pred, y)
            ssim_b, psnr_b, dists_b = all_losses(pred, y)
            
            ssim_val += ssim_b
            psnr_val += psnr_b
            dists_val += dists_b

            test_loss_sum += test_loss.item()
            psnr_score += psnr_metric(pred, y).item()

            if (batch_id + 1) % 10 == 0:
                print(
                    f"Average Test Loss till batch: {batch_id+1} = {test_loss_sum/(batch_id+1):.8f}\n"
                    f"Average PSNR score : {psnr_score/(batch_id+1):.8f}\n"
                    f"SSIM: {ssim_val/(batch_id+1):.4f} |  DISTS: {dists_val/(batch_id+1):.4f}"
                )

            
 

import torch
import albumentations as A
import numpy as np

