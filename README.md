# Bharat Atomic — AI-Driven SEM Image Enhancement Pipeline

> **Internship Project** | Tejas Shrivastava | Bharat Atomic Labs
>
> *Real-time deep learning restoration of Scanning Electron Microscope (SEM) imagery — from synthetic noise engineering to production inference.*

---

## Table of Contents

- [Bharat Atomic — AI-Driven SEM Image Enhancement Pipeline](#bharat-atomic--ai-driven-sem-image-enhancement-pipeline)
  - [Table of Contents](#table-of-contents)
  - [1. Company \& Mission](#1-company--mission)
  - [2. Why SEM Image Quality Matters](#2-why-sem-image-quality-matters)
  - [3. Project Overview \& Phases](#3-project-overview--phases)
  - [4. Repository Structure](#4-repository-structure)
  - [5. Phase 1 — SEM Image Denoising](#5-phase-1--sem-image-denoising)
    - [5.1 Dataset](#51-dataset)
    - [5.2 SEM Noise Synthesis Pipeline](#52-sem-noise-synthesis-pipeline)
      - [Noise Types](#noise-types)
      - [Composite Augmentation Strategy](#composite-augmentation-strategy)
    - [5.3 Model Architectures](#53-model-architectures)
      - [DnCNN](#dncnn)
      - [Attention U-Net](#attention-u-net)
      - [RDUNet (Fine-Tuned)](#rdunet-fine-tuned)
      - [NAFNet (Fine-Tuned)](#nafnet-fine-tuned)
    - [5.4 Loss Function Design](#54-loss-function-design)
      - [Phase 1 Loss — SSIM Composite](#phase-1-loss--ssim-composite)
      - [Phase 2 Loss — DISTS Composite (Current)](#phase-2-loss--dists-composite-current)
      - [Optimiser \& Scheduler](#optimiser--scheduler)
    - [5.5 Training Infrastructure](#55-training-infrastructure)
    - [5.6 Results \& Benchmarks](#56-results--benchmarks)
    - [5.7 Key Design Decisions](#57-key-design-decisions)
  - [6. Phase 2 — Segmentation, Deblurring \& Super-Resolution *(Planned)*](#6-phase-2--segmentation-deblurring--super-resolution-planned)
  - [7. Phase 3 — RL-Based Autonomous SEM Calibration *(Planned)*](#7-phase-3--rl-based-autonomous-sem-calibration-planned)
  - [8. Codebase Reference](#8-codebase-reference)
    - [`semPhase1/notebooks/base.py`](#semphase1notebooksbasepy)
  - [9. Installation \& Setup](#9-installation--setup)
    - [Requirements](#requirements)
    - [Pre-trained Weights](#pre-trained-weights)
  - [10. Usage](#10-usage)
    - [Training a Model](#training-a-model)
    - [Single-Image Inference](#single-image-inference)
  - [11. Metrics \& Evaluation Philosophy](#11-metrics--evaluation-philosophy)
  - [12. Computational Constraints \& GPU Roadmap](#12-computational-constraints--gpu-roadmap)
    - [Current Bottleneck: Apple M3 Max (MPS)](#current-bottleneck-apple-m3-max-mps)
    - [GPU Target: NVIDIA (Incoming)](#gpu-target-nvidia-incoming)
  - [13. Lessons Learned](#13-lessons-learned)
  - [14. Remaining Work \& Roadmap](#14-remaining-work--roadmap)
    - [Phase 1 Remaining](#phase-1-remaining)
    - [Phase 2 *(Planned)*](#phase-2-planned)
    - [Phase 3 *(Planned)*](#phase-3-planned)
  - [15. References](#15-references)

---

## 1. Company & Mission

**Bharat Atomic Labs** is a deep-tech hardware startup building production-grade **Field-Emission Scanning Electron Microscopes (FE-SEMs)** in India. An SEM is a scientific instrument that scans a focused electron beam across a sample surface, collecting secondary electrons, backscattered electrons, and X-rays to produce nanometer-resolution images of material surfaces and microstructures.

The commercial impact of SEM spans semiconductor inspection, materials science, pharmaceutical QC, academic research, and industrial failure analysis. However, the entire software stack — image processing, analysis, calibration — is currently proprietary and locked to Western vendors. **Bharat Atomic's vision is to build this stack entirely in-house, with AI-driven automation as a core differentiator.** This internship project is part of that vision.

---

## 2. Why SEM Image Quality Matters

SEM images are not photographs. They are constructed **pixel-by-pixel** by counting electron emission events at each scan position. Image quality is directly tied to how long the electron beam dwells at each point (dwell time):

- **Slow scan** → high dwell time → many electrons counted → high SNR → clean image
- **Fast scan** → low dwell time → few electrons counted → low SNR → noisy, grainy image

The fast scan is the practical default for users who want live previews or quick results. But restoring fast-scan images to slow-scan quality is non-trivial because:

1. **Slow scans are not always safe.** Prolonged high-energy electron exposure causes beam-induced damage on polymers, biologicals, and ceramics — breaking chemical bonds and altering the very surface being imaged.
2. **Slow scans can introduce their own artifacts.** High cumulative dose on non-conductive samples causes **electrostatic charging**, which produces bright streaks, image distortion, and physically repels subsequent electrons.
3. **Downstream tasks require clean images.** Particle sizing, defect detection, crystal structure classification, and segmentation model training all fail or degrade on noisy inputs.

The objective of Phase 1 is to **take a fast-scan (noisy) SEM image and restore it to the quality of a slow, careful scan** — perceptually clean, with fine surface features intact — using deep learning.

---

## 3. Project Overview & Phases

This project is structured across three phases of increasing complexity:

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | SEM Image Denoising & Restoration | ✅ In progress (experiments complete, GPU training pending) |
| **Phase 2** | Defect Segmentation, Deblurring, Super-Resolution | 🔜 Planned |
| **Phase 3** | RL-Based Autonomous SEM Parameter Calibration | 🔜 Planned |

---

## 4. Repository Structure

```
bharatAtomic/
│
├── README.md                        ← This file (parent-level project overview)
│
├── semPhase1/                       ← Phase 1: Denoising pipeline
│   ├── README.md                    ← Phase 1 detailed notebook/code README
│   ├── dataset/
│   │   └── crop/images/             ← NFFA-Europe SEM dataset (256×256 crops)
│   └── notebooks/
│       ├── base.py                  ← Shared pipeline: data, noise, training, eval
│       ├── models/                  ← Architecture definitions
│       │   ├── rdunet.py            ← RDUNet (Dense Residual U-Net)
│       │   ├── NAFNet_arch.py       ← NAFNet (Nonlinear Activation Free Network)
│       │   ├── AttentionNet.py      ← Custom Attention U-Net
│       │   └── dncnn.py             ← DnCNN (Denoising CNN baseline)
│       ├── rdunet.ipynb             ← RDUNet training & evaluation notebook
│       ├── nafnet.ipynb             ← NAFNet fine-tuning notebook
│       ├── dncnn.ipynb              ← DnCNN from-scratch training notebook
│       ├── attentionNet.ipynb       ← Attention U-Net training notebook
│       └── images/                  ← Test images for inference demos
│
├── semPhase2/                       ← Phase 2 (planned)
└── semPhase3/                       ← Phase 3 (planned)
```

---

## 5. Phase 1 — SEM Image Denoising

Phase 1 establishes the complete supervised denoising pipeline: real SEM dataset → synthetic noise generation → model training → evaluation. Four architectures were implemented, trained, and benchmarked.

### 5.1 Dataset

**Source:** [NFFA-Europe Majority SEM Dataset](https://b2nd.eudat.eu/dataset/31296749-82da-5e45-8cf1-1e710a12f4ac) — an open scientific dataset from the European Nanoscience Foundries and Fine Analysis (NFFA) infrastructure project, containing SEM micrographs from multiple institutions across a wide variety of materials: particles, biological samples, fibres, porous materials, MEMS structures, and more.

**Why real SEM images and not ImageNet?** General-purpose natural image datasets have fundamentally different statistical properties from electron micrographs — different spatial frequency distributions, no colour, different texture classes. Training on ImageNet would require additional domain adaptation. Starting from actual SEM data ensures the model learns texture, edge, and structure representations grounded in electron microscopy from the very first gradient step.

**Dataset covers all three SEM detector types:**

| Detector | Signal | Contrast Mode |
|---|---|---|
| SE (Secondary Electron) | Topographic | Surface topology |
| BSE (Backscattered Electron) | Compositional | Atomic number contrast |
| X-Ray (EDS/EDX) | Elemental | Spatial element distribution |

**Preparation:**
- Raw images cropped and resized to **256×256 pixels** (balance between microstructural detail and memory constraints; 512×512 and 1024×1024 are planned once GPU hardware is available)
- Normalised to float32 range `[0, 1]`
- RGB (3-channel) for NAFNet/Attention U-Net/DnCNN; converted to 1-channel grayscale using weighted-average greyscaling (`A.ToGray(method='weighted_average', p=1.0)`) for RDUNet
- **Split:** 70% training / 30% test, with a further 20% of training held out for validation → effective split: ~56% train / 14% val / 30% test
- Each clean image augmented with **4–5 randomly generated noisy versions** → total dataset of approximately **65,000 image pairs**

---

### 5.2 SEM Noise Synthesis Pipeline

This is the most technically novel and critical component of Phase 1. Standard Gaussian noise is insufficient — real SEM images suffer from a combination of physical effects arising from the electron beam, detector electronics, and scanning mechanism. A model trained only on Gaussian noise would fail in deployment on real instruments.

The `NoiseImage` class in `base.py` implements **six distinct noise functions**, each modelling a real physical source:

#### Noise Types

**1. Mixed Poisson-Gaussian Noise** *(applied with ~85% probability)*

Electron detection is a counting process — the number of electrons detected at each pixel follows a **Poisson distribution** (lower signal = higher relative variance). This is the dominant noise source in fast-scan images. Detector electronics add an additional Gaussian readout noise layer on top.

```
Poisson scale ∈ [32, 128]   (controls shot noise intensity)
Gaussian var  ∈ [0.002, 0.012]  (controls electronics noise floor)
```

**2. Scanline Noise / Horizontal Banding** *(applied with ~55% probability)*

The SEM scans line-by-line. Gun instability, electromagnetic interference, or ground loops cause beam current fluctuations between lines — each horizontal row gets a slightly different brightness offset, producing characteristic horizontal banding.

```
Line variance ∈ [0.004, 0.018]  (controls banding intensity)
```

**3. Line Shift Artifacts** *(applied with ~35% probability)*

Mechanical or electrical glitches cause individual scan lines to shift laterally relative to neighbours — a "tearing" appearance. Each row is independently triggered and rolled by an integer pixel offset.

```
Max shift  ∈ [3, 10] pixels
Shift prob ∈ [0.05, 0.20]  (per-row trigger probability)
```

**4. Drift Distortion** *(applied with ~40% probability)*

During long scans, thermal expansion or electrostatic effects cause the beam or sample to drift slowly. This produces **progressive lateral displacement accumulating over the image**, modelled as a cumulative random walk applied per row.

```
Max drift ∈ [4.0, 14.0] pixels
```

**5. Charging Artifact** *(applied with ~30% probability)*

When the electron beam hits a non-conductive sample, charge accumulates on the surface. This creates a slowly varying **low-frequency intensity gradient** across the image — a characteristic "glow" or "darkening" — modelled as a linear gradient field with slight random wobble.

```
Strength ∈ [0.15, 0.45]   (gradient magnitude)
Orientation: vertical or horizontal (random)
```

**6. Detector Streak Pixels** *(applied with ~20% probability)*

Physical damage or electronic faults produce isolated pixels that are always saturated (hot) or always zero (dead).

```
Amount ∈ [0.0005, 0.002]  (fraction of total pixels affected)
```

#### Composite Augmentation Strategy

A critical design decision was moving from **single-noise-type augmentation** to **probabilistic composite augmentation**. In reality, a single fast-scan SEM image experiences all of these effects simultaneously. Models trained on single-noise-type images produced ring artifacts and colour blotches when tested on composite-noise images.

The `new_augment_sem` function applies each noise type **independently with its own probability**, magnitudes drawn from ranges rather than fixed values. No two augmented versions of the same image look exactly alike. This significantly improved test-set generalisation and eliminated composite-noise artifacts entirely.

> **Note:** Geometric augmentations (flips, rotations, affine transforms) were intentionally deferred. For denoising, geometric augmentations must be applied **identically** to both the noisy input and clean target (paired augmentation). Applying them independently gives the network contradictory supervision signals. Paired geometric augmentation is planned as a targeted lever once current models reach a performance plateau.

---

### 5.3 Model Architectures

Four architectures were evaluated, covering a spectrum from classical CNN baseline to state-of-the-art pretrained restoration network:

| Model | Type | Parameters | Training Strategy |
|---|---|---|---|
| **DnCNN** | Residual CNN | ~558K | From scratch |
| **Attention U-Net** | U-Net + Attention Gates | ~7.8M | From scratch |
| **RDUNet** | Dense Residual U-Net | ~12M | Fine-tune (decoder only) |
| **NAFNet (width=32)** | Nonlinear-free restoration | ~6.8M | Fine-tune (decoder + last 6 middle blocks) |

#### DnCNN

DnCNN (Zhang et al., 2017) reframes denoising as **residual learning** — instead of outputting a clean image directly, the network learns to predict the noise itself. Clean output = input − predicted noise. Architecture: one Conv+ReLU input layer, 15 intermediate Conv+BatchNorm+ReLU layers, and a final convolutional output. Trained entirely from scratch on SEM data as the baseline.

#### Attention U-Net

Custom implementation with a 3-level encoder (64→128→256 channels), 512-channel bottleneck, and 3-level decoder. Each decoder block applies an **attention gate** to its skip connection — a soft spatial mask learned from the decoder feature map that decides which encoder regions are relevant. In practice, this focuses computation on textured regions that need denoising while suppressing smooth, featureless areas. Trained from scratch.

#### RDUNet (Fine-Tuned)

RDUNet (Gurrola-Ramos et al., 2021) combines U-Net's skip-connection hierarchy with **densely connected internal blocks** — each block concatenates its own intermediate outputs into a running feature stack before producing its final output, creating rich gradient highways during backpropagation. PReLU activations add learnable per-channel non-linearity.

Pre-trained weights (`model_gray.pth`) were loaded from the original authors' public GitHub. **Frozen layers:** full encoder path through bottleneck + first decoder block at level 2. **Trainable:** remaining decoder blocks and output block. This prevents catastrophic forgetting of pretrained features while adapting the reconstruction head to SEM imagery.

> RDUNet operates on **1-channel grayscale** input — requiring the weighted-average greyscale conversion step in the transform pipeline.

#### NAFNet (Fine-Tuned)

NAFNet (Chen et al., 2022) eliminates expensive non-linear operations — replacing standard channel attention with a **SimpleGate** (splits channels, multiplies two halves) and removing ReLU/GELU entirely. Despite apparent simplicity, this achieves state-of-the-art performance on the SIDD denoising benchmark and is notably faster than prior architectures.

Pre-trained checkpoint: `NAFNet-SIDD-width32.pth` (trained on smartphone camera noise). **Frozen:** full encoder + first half of middle blocks. **Trainable:** `middle_blks[6–11]`, `decoders[1–3]`, `ups[1–3]`. The encoder captures general texture/noise priors transferable across domains; the decoder needs domain adaptation for SEM-specific reconstruction.

---

### 5.4 Loss Function Design

#### Phase 1 Loss — SSIM Composite

Initial training used a composite of pixel accuracy and structural similarity:

```
Loss = (1 − θ) × L1(pred, target) + θ × (1 − SSIM(pred, target))
```

where θ = 0.4. L1 ensures pixel-level accuracy; `(1 − SSIM)` encourages structural similarity. SSIM evaluates luminance, contrast, and local structure, closer to how human vision perceives images than per-pixel MSE.

#### Phase 2 Loss — DISTS Composite (Current)

After reviewing the DISTS paper (Ding et al., 2021), which showed that DISTS scores are better correlated with human perceptual judgements than either PSNR or SSIM — especially for texture-rich images — the loss was switched to:

```
Loss = (1 − θ) × L1(pred, target) + θ × DISTS(pred, target)
```

The key insight: SSIM and PSNR measure fidelity at specific pixel locations; **DISTS measures structural and textural pattern similarity in a shift-invariant, multi-scale way** using a pre-trained VGG feature extractor. For SEM images, surface texture is often the most scientifically important feature. This switch produced **visibly sharper grain boundaries and clearer surface features**, even when the PSNR improvement was marginal (<0.5 dB).

> Note: Unlike `(1 − SSIM)`, the DISTS score is a raw distance (lower = more similar) and enters the loss directly, not subtracted from 1.

Experiments were run with θ = 0.4 and θ = 0.75. Higher θ biases the loss toward perceptual quality, potentially at the cost of pixel-level accuracy.

#### Optimiser & Scheduler

All models trained with **Adam** (lr = 1e-4, default betas). A **ReduceLROnPlateau** scheduler halved the learning rate after 5 epochs of no improvement in validation loss, with a minimum lr floor of 1e-8. This adaptive scheduling was critical given limited epoch counts — preventing wasted epochs at learning rates that were too high or too low.

---

### 5.5 Training Infrastructure

**Hardware:** MacBook Pro, Apple M3 Max chip (38-core GPU, MPS backend). All training via PyTorch's Metal Performance Shaders backend (`torch.device('mps')`).

**Constraints:**
- No CUDA — the most optimised PyTorch training kernels are CUDA-only; MPS is functional but less mature
- Unified memory — GPU and CPU share the same pool; batch sizes above 4 at 256×256 cause memory pressure
- No TensorRT — NVIDIA's inference optimisation runtime is CUDA-only
- Long epoch times: 90 minutes to 8–9 hours depending on architecture and dataset size

**Mixed Precision:** `torch.bfloat16` via `torch.amp.autocast(device_type="mps", dtype=torch.bfloat16)` — halves activation memory footprint while retaining float32 for loss and backward pass. Provides ~15–20% speedup on MPS.

**Checkpointing:** Saves every `save_freq` epochs (typically 1–2) to avoid losing multi-hour runs to thermal throttling or interruptions. Running best-model checkpoint maintained on validation loss.

**Batch size:** 4 throughout (MPS memory pressure constraint).

---

### 5.6 Results & Benchmarks

All results are from the best training run for each architecture on the 30% holdout SEM test set (256×256 resolution, NFFA-Europe dataset), achieved within **5–10 fine-tuning epochs** on the MPS hardware:

| Model | PSNR (dB) | SSIM | Val Loss | Inference Time (MPS) |
|---|---|---|---|---|
| DnCNN (from scratch) | 26–30 | 0.91–0.97 | 0.02–0.04 | ~110 ms |
| Attention U-Net (scratch) | 24–27 | 0.93–0.95 | 0.015–0.03 | TBD |
| RDUNet (fine-tuned) | 27–38 | 0.93–0.98 | 0.01–0.03 | TBD |
| NAFNet (fine-tuned) | 27–34 | 0.95–0.98 | 0.01–0.02 | ~70 ms |

**Headline gains (averaged across all models):**
- PSNR improvement: **+3 to +5 dB** over degraded input (approx. 15–22% perceptual improvement)
- DISTS score improvement: **40–60%**
- Lowest single-image inference: **~60 ms** | Typical range: **120–200 ms** (MPS)

**Key observations:**

- **RDUNet** achieved the highest peak PSNR (up to 38 dB) — the dense connectivity and skip structure are highly effective for structural recovery
- **NAFNet** offered the best trade-off between quality and speed (~70 ms, 27–34 dB PSNR) — the elimination of expensive non-linearities pays off in latency
- **Initial MVP target** was 26–28 dB. Both fine-tuned models already exceed this significantly
- **Fine-tuned models** (NAFNet, RDUNet) significantly outperformed from-scratch models on SEM-specific structured artifacts (scanline banding, charging gradients). From-scratch models were better at isotropic shot noise but struggled with directional, correlated artifacts — validating the decision to transfer pretrained priors
- **DISTS composite loss** produced visibly sharper texture preservation compared to SSIM composite, even at similar PSNR — grain boundaries and surface features were noticeably cleaner

**Inference target:** The real-time deployment goal is <40–60 ms per frame at **1K resolution** at 16–24 fps. NAFNet at 256×256 achieves ~70 ms on MPS. At 1K (16× more pixels), direct scaling is not linear due to parallelism, but sub-40 ms is unlikely on MPS without CUDA+TensorRT. On an RTX 4090 with TensorRT, NAFNet at 1K could realistically achieve **15–25 ms**, meeting the real-time target.

> **Important caveat:** All metrics are measured on synthetically noised images where the ground truth is exactly known. This tends to inflate numbers compared to deployment on real fast-scan hardware images. True performance on live SEM hardware remains to be validated — this is a key remaining Phase 1 deliverable.

---

### 5.7 Key Design Decisions

| Decision | Rationale | Outcome |
|---|---|---|
| Use NFFA-Europe SEM dataset (not ImageNet) | Domain alignment: natural images have different statistics | Better generalisation to real SEM test images |
| Composite SEM noise (not single Gaussian) | Real SEM images have simultaneous multiple noise sources | Eliminated composite-noise test artifacts; better real-world generalisation |
| Fine-tune NAFNet/RDUNet instead of training from scratch | Limited compute; pretrained weights carry strong texture/edge priors | Faster convergence, better metrics per compute hour |
| Freeze encoder, train decoder for fine-tuning | Encoder features are domain-agnostic; decoder reconstruction is domain-specific | Prevented catastrophic forgetting of pretrained features |
| Switch from SSIM composite to DISTS composite loss | DISTS better correlates with human perceptual judgement; SEM textures are scientifically critical | Visibly sharper texture and grain boundary preservation |
| Reject pix2pixHD (GAN architecture) | GAN hallucination is unacceptable in scientific imaging — cannot invent surface features | Maintained physical fidelity; regression-based models are conservative by design |
| 256×256 patches instead of full resolution | Memory constraints on MPS | Feasible training; to be revisited with GPU hardware |
| Batch size = 4 | MPS memory pressure above batch 4 at 256×256 | Accepted training noise; compensated with LR scheduler |

---

## 6. Phase 2 — Segmentation, Deblurring & Super-Resolution *(Planned)*

Phase 2 begins after Phase 1 sign-off and builds directly on the restored images produced by the Phase 1 denoiser.

**Defect Detection & Feature Segmentation:**
Training a segmentation model (likely **YOLOv8-seg** combined with **SAM2** for instance segmentation) to identify and localise surface features, defects, grain boundaries, and particles in restored SEM images. Phase 1 denoising is a prerequisite — segmentation models fail or degrade on noisy inputs.

**Deblurring & Resolution Enhancement:**
SEM images acquired at high magnification or with beam focus issues can appear blurry rather than noisy. Deblurring handles deterministic blur; **super-resolution** aims to recover sub-pixel detail from lower-magnification images. These are related but distinct from denoising and require dedicated model variants.

**Multimodal Input:**
The system will eventually handle SE, BSE, and EDS (X-Ray) images together, as each carries complementary information about the sample. Phase 2 will begin exploring multi-channel input architectures.

---

## 7. Phase 3 — RL-Based Autonomous SEM Calibration *(Planned)*

Phase 3 is the most technically ambitious component: a **closed-loop reinforcement learning agent** that observes the current image quality and automatically adjusts SEM operating parameters — beam current, accelerating voltage, focus, stigmation, working distance — to maximise image quality without human operator input.

**RL Formulation:**
- **State:** Current (noisy/blurry) SEM image + quality metrics (PSNR, SSIM, DISTS)
- **Action space:** Adjustable SEM parameters (beam current, accelerating voltage, focus, stigmation, working distance)
- **Reward:** Improvement in image quality after parameter change

The agent must learn not just which parameters to change, but how much and in what sequence — parameter interactions in SEMs are non-trivial and can be non-monotonic.

---

## 8. Codebase Reference

### `semPhase1/notebooks/base.py`

The central shared module. All model notebooks import from this file.

| Component | Description |
|---|---|
| `list_images(parent_path)` | Recursively collects `.jpg` images from directory tree |
| `collect_images(parent_path)` | Wrapper for `list_images`, prints dataset stats |
| `load_all_data(parent_dir, split, times)` | End-to-end loader: collects images, creates `CustomData` train/test splits, returns DataLoaders |
| `transform_3(size)` | Returns `A.Compose([Resize(size)])` — RGB pipeline |
| `transform_1(size)` | Returns `A.Compose([Resize(size), ToGray()])` — grayscale pipeline for RDUNet |
| `NoiseImage` | Full SEM noise simulator — 6 noise functions + `new_augment_sem` composite |
| `CustomData(Dataset)` | PyTorch Dataset: applies transform, generates noisy input from clean target on-the-fly |
| `load_data(...)` | Splits train dataset into train/val subsets, creates DataLoaders |
| `calc_loss(pred, target, metric, theta)` | Weighted L1 + SSIM or DISTS composite loss |
| `all_losses(pred, target, train, c)` | Returns (SSIM, PSNR, DISTS) tuple for evaluation |
| `fineTune(...)` | Full training loop: forward pass, backprop, scheduler, validation, checkpointing, plotting |
| `test_func(model, ip_img, transform, channels, device)` | Single-image inference: load → resize → noise → infer → metrics → visualise |
| `test_func_batches(model, test_loader, device)` | Batched evaluation over a DataLoader |

> **Important usage note:** `transform_3` and `transform_1` are **factory functions** — they must be called with a size argument to obtain the `A.Compose` object. Pass the result, not the function itself:
>
> ```python
> # ✅ Correct
> test_func(model, image_path, transform=transform_1(256), channels=1, device=device)
>
> # ❌ Wrong — passes the function, not the composed transform
> test_func(model, image_path, transform=transform_1, channels=1, device=device)
> ```

---

## 9. Installation & Setup

### Requirements

```bash
# Core
torch torchvision torchaudio          # PyTorch (MPS or CUDA)
albumentations                        # Image augmentation
numpy pillow tifffile matplotlib

# Metrics
pytorch-msssim                        # SSIM loss
torchmetrics                          # PSNR metric
DISTS-pytorch                         # DISTS perceptual metric

# Architectures
# NAFNet: NAFNet_arch.py, local_arch.py, arch_util.py (included in models/)
# RDUNet: models/rdunet.py (included)
```

```bash
pip install torch torchvision albumentations pillow tifffile matplotlib \
            pytorch-msssim torchmetrics DISTS-pytorch
```

### Pre-trained Weights

| Model | Checkpoint | Source |
|---|---|---|
| NAFNet | `NAFNet-SIDD-width32.pth` | [Original NAFNet repo](https://github.com/megvii-research/NAFNet) |
| RDUNet | `model_gray.pth` | [Gurrola-Ramos et al. GitHub](https://github.com/jgurramCR/RDUNet) |

Place checkpoints in `semPhase1/notebooks/models/checkpoints/` and update paths in each notebook before loading.

---

## 10. Usage

### Training a Model

Open the relevant notebook (`nafnet.ipynb`, `rdunet.ipynb`, `dncnn.ipynb`, `attentionNet.ipynb`) and follow the cells in order. The shared pipeline is imported from `base.py`:

```python
from base import load_all_data, fineTune, transform_1, transform_3

# Load data
train_loader, val_loader, test_loader = load_all_data(
    parent_dir='../dataset/crop/images',
    split=0.7,
    times=5
)

# Fine-tune
fineTune(
    model=nafnet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    theta=0.4,
    metric='dists',
    name='nafnet_sem_v1.pth',
    device=device
)
```

### Single-Image Inference

```python
from base import test_func, transform_3, transform_1

# For NAFNet (RGB, 3-channel)
test_func(
    model=nafnet_model,
    ip_img='/path/to/sem_image.jpg',
    transform=transform_3(256),        # ← call with size first
    channels=3,
    device=device
)

# For RDUNet (Grayscale, 1-channel)
test_func(
    model=rdunet_model,
    ip_img='/path/to/sem_image.jpg',
    transform=transform_1(256),        # ← call with size first
    channels=1,
    device=device
)
```

The function outputs:
```
Loss: 0.01234567 | SSIM: 0.9712
PSNR: 32.4821 dB | DISTS: 0.0431
Initial PSNR: 28.1200 | Initial DISTS: 0.1023
Improvement: +4.36 dB
Pred Time: 0.0712s | Total Time: 0.0843s
```
followed by a side-by-side visualisation of input (degraded), prediction, and label (clean).

---

## 11. Metrics & Evaluation Philosophy

Three complementary metrics are tracked across all experiments:

| Metric | What It Measures | Why It's Used |
|---|---|---|
| **PSNR** (Peak Signal-to-Noise Ratio, dB) | Pixel-level reconstruction accuracy | Standard reference; higher is better |
| **SSIM** (Structural Similarity Index) | Perceived luminance, contrast, and local structure | Closer to human vision than MSE |
| **DISTS** (Deep Image Structure & Texture Similarity) | Multi-scale structural + textural similarity via VGG features | Best correlation with human perceptual judgement, especially for textures |

PSNR and SSIM measure fidelity at specific pixel locations. DISTS is shift-invariant and captures whether the restored image has the same kinds of textures and structures as the clean target, even if they are not pixel-perfectly aligned. For SEM images, where **surface texture is the primary scientific observable**, DISTS is the most meaningful metric for deployment quality.

Improvements of less than 0.5 dB PSNR are perceptually invisible, but visual inspection revealed that models with marginally lower PSNR sometimes produced clearly better-looking grain boundaries. DISTS partially closes this gap. **Visual QA with domain experts remains essential** alongside quantitative metrics.

---

## 12. Computational Constraints & GPU Roadmap

### Current Bottleneck: Apple M3 Max (MPS)

| Constraint | Impact |
|---|---|
| No CUDA | Most optimised PyTorch kernels unavailable; MPS is functional but not mature |
| Unified memory (GPU + CPU shared) | Batch size capped at 4 for 256×256 inputs |
| No TensorRT | Real-time inference optimisation unavailable |
| Epoch time: 90 min – 9 hours | Very limited hyperparameter exploration; ~2–3 epochs per day |

### GPU Target: NVIDIA (Incoming)

Once a dedicated CUDA-capable GPU is available:

- Epoch times drop from 8–9 hours → **40–120 minutes**
- Batch sizes can scale to 16–32+
- Resolution can scale to 512×512 and 1024×1024
- TensorRT optimisation becomes available for production inference
- **Projected NAFNet inference on RTX 4090 + TensorRT at 1K resolution: 15–25 ms** — meeting the real-time target of <40 ms

---

## 13. Lessons Learned

**Domain understanding precedes model selection.** The highest-impact work in Phase 1 was not architecture engineering — it was understanding SEM noise physics well enough to build a realistic noise simulator. A state-of-the-art architecture trained on the wrong noise distribution would fail in deployment. The `NoiseImage` class is the intellectual core of Phase 1.

**Metrics and perception diverge.** PSNR improvements below 0.5 dB are perceptually invisible. DISTS partially closes the gap between objective scores and visual quality, but no metric fully replaces domain expert visual QA.

**Pretrained models are a force multiplier under compute constraints.** With epoch times of 8–9 hours on the current hardware, the ability to start from a pretrained checkpoint and reach strong validation metrics in 20–30 epochs (instead of 150+) was critical. Fine-tuning was not just a shortcut — it was the correct engineering decision.

**Composite noise training is non-negotiable for real-world deployment.** Single-noise-type training produces models that fail on real images. The switch to probabilistic composite noise augmentation (`new_augment_sem`) eliminated composite-noise test artifacts and is the single change most responsible for improved generalisation.

**GAN hallucination is a dealbreaker for scientific instruments.** pix2pixHD was briefly considered and rejected. A denoising model that invents surface features that were not present — even if they look plausible — is unacceptable in a measurement instrument context. Regression-based models with perceptual losses are conservative by design.

---

## 14. Remaining Work & Roadmap

### Phase 1 Remaining

- [ ] **GPU-scale training** — full training runs on NVIDIA hardware with optimised batch sizes, higher resolution, and extended epochs
- [ ] **Real SEM image validation** — inference on actual fast-scan images from Bharat Atomic's hardware; qualitative validation from the technical team
- [ ] **Production inference script** — clean standalone script taking image path as input, producing restored output; no notebook dependency
- [ ] **Per-model comparative analysis** — side-by-side visual comparison of all four models on identical test images with per-image metrics

### Phase 2 *(Planned)*

- [ ] Defect detection and feature segmentation (YOLOv8-seg + SAM2)
- [ ] Deblurring model for focus-artifact correction
- [ ] Super-resolution for sub-pixel detail recovery
- [ ] Multimodal SE + BSE + EDS input fusion

### Phase 3 *(Planned)*

- [ ] RL formulation for autonomous SEM parameter calibration
- [ ] State encoder from Phase 1 denoiser + Phase 2 segmenter
- [ ] Simulation environment for RL training before hardware deployment
- [ ] Closed-loop testing on live SEM system

---

## 15. References

1. Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). Image denoising by sparse 3-D transform-domain collaborative filtering. *IEEE Transactions on Image Processing*, 16(8), 2080–2095.
2. Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. *IEEE Transactions on Image Processing*, 26(7), 3142–3155.
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*, LNCS 9351, 234–241.
4. Oktay, O., et al. (2018). Attention U-Net: Learning where to look for the pancreas. *MIDL 2018*. arXiv:1804.03999.
5. Chen, L., Chu, X., Zhang, X., & Sun, J. (2022). Simple baselines for image restoration. *ECCV 2022*. arXiv:2204.04676.
6. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612.
7. Ding, K., Ma, K., Wang, S., & Simoncelli, E. P. (2021). Comparison of full-reference image quality models for optimization of image processing systems. *International Journal of Computer Vision*, 129(4), 1258–1281.
8. Aversa, R., et al. (2018). The first annotated set of scanning electron microscopy images for nanoscience. *Scientific Data*, 5, 180172. [NFFA-Europe Majority SEM Dataset](https://b2nd.eudat.eu/dataset/31296749-82da-5e45-8cf1-1e710a12f4ac).
9. Lehtinen, J., et al. (2018). Noise2Noise: Learning image restoration without clean data. *ICML 2018*. arXiv:1803.04189.
10. Buslaev, A., et al. (2020). Albumentations: Fast and flexible image augmentations. *Information*, 11(2), 125.
11. Gurrola-Ramos, J., Dalmau, O., & Alarcón, T. E. (2021). A residual dense U-Net neural network for image denoising. *IEEE Access*, 9, 31742–31754.

---

all weights and interactive models stored in google drive (https://drive.google.com/drive/folders/1rnN5juQu7kbgzvjLacVFdwx_7DZAfzIc?usp=sharing)