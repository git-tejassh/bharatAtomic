# Bharat Atomic SEM Denoising Experiments

This repository documents an image restoration pipeline built for **SEM-style image denoising** using multiple deep learning architectures. The current setup focuses on generating realistic synthetic SEM degradation, training restoration models on paired noisy-clean data, and evaluating whether lightweight fine-tuning can quickly recover image quality in terms of PSNR, SSIM, and DISTS [file:1].

The project structure is centered around a shared experimentation layer in `base.py`, with model-specific notebooks for architectures such as RDUNet, NAFNet, DnCNN, and Attention U-Net referenced by the workflow. The common code handles dataset creation, domain-aware noise synthesis, transforms, dataloading, training, validation, inference testing, and metric reporting so that different denoising models can be compared under a similar pipeline [file:1].

## Project Goal

The main objective is to train and compare several image denoising networks for SEM-like microscopy imagery. Instead of relying only on generic corruption, the code introduces SEM-relevant artifacts such as mixed Poisson-Gaussian noise, scanline noise, line-shift artifacts, drift distortion, charging artifacts, and detector streak pixels, which makes the training setup closer to real microscope degradation patterns [file:1].

The intent is not just to remove visible noise, but also to preserve structural detail and perceptual similarity. That is why the pipeline tracks not only PSNR and SSIM, but also DISTS, a perceptual similarity metric that is useful for judging whether restored outputs remain faithful to the underlying microstructure [file:1].

## Core Workflow

The repository follows a paired denoising workflow. First, clean images are collected from a directory tree, converted to arrays, shuffled, and split into training and testing subsets through `load_all_data`; then a `CustomData` dataset applies resizing transforms and generates noisy inputs from clean targets using the SEM noise engine [file:1].

A key design choice in the dataset pipeline is that the clean image is transformed first, and noise is added afterward. This preserves alignment between noisy input and clean target while ensuring the degradation process reflects the resized training domain used by each model [file:1].

## Data Pipeline

`base.py` loads `.jpg` images recursively, converts them to arrays, and builds train, validation, and test loaders using `torch.utils.data.Dataset`, `DataLoader`, and `random_split`. The default shared training setup resizes images to 256 by 256 and uses repeated sampling during training so that the same base image can be seen under different noisy realizations across epochs [file:1].

Two Albumentations transform builders are defined: `transform_3(size)` for RGB pipelines and `transform_1(size)` for grayscale pipelines. This is important because some models are trained as 3-channel restorers while others, such as RDUNet in this workflow, are intended for 1-channel grayscale SEM inputs [file:1].

## SEM Noise Model

One of the most important contributions in this codebase is the handcrafted SEM noise simulator. The `NoiseImage` class includes separate functions for Gaussian noise, salt-and-pepper noise, speckle noise, Poisson noise, mixed Poisson-Gaussian noise, scanline noise, line-shift artifacts, drift distortion, charging artifacts, and detector streak pixels [file:1].

The more advanced routine, `new_augment_sem`, composes these corruptions probabilistically and returns both the noisy sample and metadata about which artifacts were applied. This means the project does not rely on a single simplistic corruption model; instead, it creates diverse SEM-like degradations that are better suited for robust denoiser training [file:1].

## Training Strategy

Model training is orchestrated through the `fineTune` function. It uses Adam optimization, a ReduceLROnPlateau scheduler, optional mixed precision on Apple MPS, checkpoint saving, epoch-wise validation, and loss computation based on a weighted combination of L1 reconstruction loss and either SSIM-based or DISTS-based terms [file:1].

This setup is useful because it balances pixel accuracy with perceptual or structural fidelity. In practice, the code logs validation loss, PSNR, SSIM, and DISTS over time, which makes it easier to judge whether a model is merely smoothing the image or actually improving meaningful structure recovery [file:1].

## Evaluation and Inference

Two testing utilities are provided. `test_func` performs single-image inference by loading an image, resizing it, applying SEM noise, running the selected model, and then printing the restored image metrics along with inference latency and PSNR improvement over the degraded input; `test_func_batches` extends this logic to test loaders for batched evaluation [file:1].

This is especially helpful for model comparison because the same interface can be reused across architectures. The function also visualizes input, prediction, and clean target side by side, which supports quick qualitative inspection in notebooks alongside the quantitative metrics [file:1].

## Models Explored

The project experimentation, as described in the notebooks requested for this README, covers multiple denoising families:

- **RDUNet** for grayscale-oriented restoration and strong structural denoising.
- **NAFNet** as a modern efficient restoration architecture.
- **DnCNN** as a classical convolutional denoiser baseline.
- **Attention U-Net / AttentionNet** for encoder-decoder denoising with attention-enhanced feature selection.

The shared message across these experiments is that all of the tested models are considered viable for final training under this pipeline, with each architecture being integrated into the same overall train-validate-test framework through the common utilities in `base.py` [file:1].

## Reported Results

According to the current experiment summary provided for this README, all of the tried models are performing strongly enough to be considered for final training. The reported gains are an average PSNR improvement of about 3 to 5 dB, approximately 15 to 22 percent improvement within only 5 to 10 epochs, DISTS improvement in the range of 40 to 60 percent, and inference times reaching as low as about 60 ms with a more typical range around 120 to 200 ms per sample [file:1].

These early-stage results suggest that the pipeline is both effective and computationally practical. A relatively small number of fine-tuning epochs already produces measurable restoration gains, which is a good sign for scaling the experiments toward longer training runs, better checkpoint selection, and model-specific optimization. 

## What Has Been Done

The following major components are already implemented in the current repository state:

- Image collection and recursive dataset loading from SEM image folders [file:1].
- Shared RGB and grayscale transform builders using Albumentations [file:1].
- Domain-specific SEM noise simulation rather than only generic synthetic noise [file:1].
- Paired noisy-clean training dataset generation with repeated sampling [file:1].
- Unified fine-tuning loop with validation, checkpointing, and scheduler support [file:1].
- Support for multiple evaluation metrics: L1-based objective, SSIM, PSNR, and DISTS [file:1].
- Single-image and batch-level inference utilities with metric reporting and visualization [file:1].
- Multi-model experimentation across RDUNet, NAFNet, DnCNN, and Attention U-Net style architectures, as stated in the notebook workflow and experiment summary [file:1].

## Why This Matters

SEM imagery often contains structured artifacts that are different from ordinary natural-image noise. Because this project explicitly simulates microscope-style corruption and evaluates perceptual fidelity, it is closer to a deployment-oriented denoising pipeline than a generic academic image restoration demo [file:1].

The current results indicate that the pipeline already delivers meaningful quality gains at relatively low training cost. That combination of realistic degradation modeling, fast fine-tuning, and low-latency inference makes this a strong foundation for selecting the best model for final training and eventual production use.

## Suggested Repository Notes

For long-term maintainability, the next improvement should be to cleanly organize the repository into modules for data, models, training, evaluation, and notebooks. It would also help to add per-model experiment logs, checkpoint naming conventions, and a final benchmark table so that future readers can compare RDUNet, NAFNet, DnCNN, and Attention U-Net using one consistent scorecard [file:1].

Another useful next step is to save representative qualitative outputs for each model under the same noisy inputs. That would make the README even stronger by showing not just the metric improvements, but also how each architecture behaves visually on SEM textures, edges, and artifact-heavy regions.
