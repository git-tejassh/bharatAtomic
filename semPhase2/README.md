# Bharat Atomic SEM Phase 2 — Open-Set Instance Segmentation & Microstructure Analysis

> **Phase 2 of the AI-Driven SEM Image Enhancement Pipeline** | Bharat Atomic Labs
>
> *From denoised images to quantitative microstructure — open-set binary instance segmentation across arbitrary SEM sample materials, with downstream morphometric analysis (particle size distribution, aspect ratio, circularity, porosity).*

---

## Table of Contents

- [1. Phase 2 in Context](#1-phase-2-in-context)
- [2. The Core Problem — What "Open-Set" Actually Means for a Product SEM](#2-the-core-problem--what-open-set-actually-means-for-a-product-sem)
- [3. Binary Segmentation Strategy — The Decision That Unlocks Everything](#3-binary-segmentation-strategy--the-decision-that-unlocks-everything)
  - [3.1 Why Binary?](#31-why-binary)
  - [3.2 What Binary Buys Us Right Now](#32-what-binary-buys-us-right-now)
  - [3.3 The Upgrade Path](#33-the-upgrade-path)
- [4. FiLM-Based Metadata Conditioning](#4-film-based-metadata-conditioning)
  - [4.1 The OHE Scaling Problem and Why We Don't Use It](#41-the-ohe-scaling-problem-and-why-we-dont-use-it)
  - [4.2 Learned Embedding Table — The Solution](#42-learned-embedding-table--the-solution)
  - [4.3 The 10 NFFA Material Classes as Initial Conditioning Labels](#43-the-10-nffa-material-classes-as-initial-conditioning-labels)
  - [4.4 Auto-Classification Safety Net](#44-auto-classification-safety-net)
- [5. Model Architecture Candidates](#5-model-architecture-candidates)
  - [5.1 Primary Track — LBMS-SAM (Frozen SAM + GSEFE + MDFF)](#51-primary-track--lbms-sam-frozen-sam--gsefe--mdff)
  - [5.2 Secondary Track — RF-DETR](#52-secondary-track--rf-detr)
  - [5.3 Secondary Track — YOLOv11-seg](#53-secondary-track--yolov11-seg)
  - [5.4 Baseline — Attention U-Net with FiLM Conditioning](#54-baseline--attention-u-net-with-film-conditioning)
  - [5.5 Lightweight Alternatives — MobileSAM / FastSAM](#55-lightweight-alternatives--mobilesam--fastsam)
  - [5.6 Zero-Training Booster — SAM-I-Am Style Post-Processing](#56-zero-training-booster--sam-i-am-style-post-processing)
  - [5.7 Architecture Comparison Table](#57-architecture-comparison-table)
- [6. Dataset Strategy](#6-dataset-strategy)
  - [6.1 What NFFA-Europe Gives Us and What It Doesn't](#61-what-nffa-europe-gives-us-and-what-it-doesnt)
  - [6.2 Our Real Starting Point — Zero Labeled Masks](#62-our-real-starting-point--zero-labeled-masks)
  - [6.3 Annotation Pipeline — µSAM-Assisted Bootstrapping](#63-annotation-pipeline--sam-assisted-bootstrapping)
  - [6.4 Phase 1 Noise Simulator as a Free Augmentation Engine](#64-phase-1-noise-simulator-as-a-free-augmentation-engine)
  - [6.5 Annotation Scale Targets](#65-annotation-scale-targets)
- [7. Loss Function Design for Binary Segmentation](#7-loss-function-design-for-binary-segmentation)
- [8. Training Strategy](#8-training-strategy)
  - [8.1 Freeze Strategy](#81-freeze-strategy)
  - [8.2 Training Order — Do This, Not That](#82-training-order--do-this-not-that)
  - [8.3 Self-Supervised Pretraining — What's Worth Doing and What Isn't](#83-self-supervised-pretraining--whats-worth-doing-and-what-isnt)
- [9. Evaluation Metrics](#9-evaluation-metrics)
- [10. Latency Budget and Deployment Reality](#10-latency-budget-and-deployment-reality)
- [11. Phase 2.1 — Morphometric Analysis Pipeline](#11-phase-21--morphometric-analysis-pipeline)
- [12. Repository Structure (Planned)](#12-repository-structure-planned)
- [13. Key Design Decisions](#13-key-design-decisions)
- [14. Checklist](#14-checklist)
- [15. Remaining Work and Roadmap](#15-remaining-work-and-roadmap)
- [16. References](#16-references)

---

## 1. Phase 2 in Context

Phase 1 delivered a complete SEM denoising pipeline — from raw fast-scan noisy input to a restored image with +3 to +5 dB PSNR improvement, using NAFNet and RDUNet as the primary models. Phase 2 builds directly on that output. Every image that enters the segmentation pipeline in Phase 2 is first passed through the Phase 1 denoiser — segmentation models are sensitive to noise in ways denoisers are not, and we already demonstrated in Phase 1 that noisy SEM images have different statistical properties than natural images. Segmenting on denoised images is a non-negotiable precondition, not an optional extra.

What Phase 2 is trying to do, plainly stated: take a restored SEM image, identify every distinct object or structure in it (particle, grain, pore, fibre, device structure, cell — whatever the sample contains), draw a precise mask around each one, and hand those masks to a downstream measurement pipeline that computes physical quantities about each object. Do this fast enough to be useful in an interactive microscopy session. Do it for any sample material a customer might put under the beam, not just a predefined catalog of known materials.

That last sentence — "any sample material" — is the constraint that drives nearly every architecture and dataset decision in this phase.

---

## 2. The Core Problem — What "Open-Set" Actually Means for a Product SEM

Bharat Atomic is not building a single-purpose inspection tool for one material system. It is building a general-purpose FE-SEM that customers across semiconductor inspection, materials science, pharmaceutical QC, academic research, and industrial failure analysis will use on samples we have never seen. A researcher studying a new battery chemistry, a pharma company inspecting particle coatings, a university lab imaging bacteria, a foundry examining metal grain structure — all of them will image completely different materials, and all of them will expect the AI to work.

This rules out the obvious approach of training a closed-set detector like YOLO or RF-DETR on a fixed list of material classes and shipping that as the primary segmentation system. A closed-set model that knows about particles, fibres, and MEMS devices will produce confidently wrong or empty outputs when a customer puts in something it has never seen. There is no warning, no graceful degradation — it simply fails silently.

The solution is an **open-set segmenter** — a model whose fundamental prior is "find the boundaries between distinct objects and background" rather than "find objects of class X." SAM (Segment Anything Model) and its derivatives are exactly this. SAM was trained on over a billion masks across tens of millions of images and has learned a general, material-agnostic concept of "object boundary." Its zero-shot output on a completely new material type degrades gracefully — it may miss some boundaries or merge touching objects, but it does not hallucinate class labels or fail catastrophically.

This is the architectural starting point for Phase 2.

---

## 3. Binary Segmentation Strategy — The Decision That Unlocks Everything

### 3.1 Why Binary?

The NFFA-Europe dataset, which is our only current labeled resource, has image-level class labels — it tells us "this whole image is particles" or "this whole image is MEMS" — but has zero pixel-level annotations. We are starting Phase 2 with no segmentation masks at all. The annotation effort required to produce multi-class semantic masks (grain / pore / secondary phase / background for ceramics; nucleus / cytoplasm / membrane / background for biological samples) is enormous and requires material-specific domain expertise at every step.

Binary segmentation — foreground object versus background — sidesteps all of this. The annotation task becomes: "draw a mask around each distinct object." No class taxonomy needed. No material-specific label definitions needed. A µSAM-assisted annotator can do this without being a domain expert in ceramics or biochemistry, because the question is purely geometric: "is this pixel part of an object or part of the background?"

This is not a compromise — it is the correct scope for Phase 2. The downstream morphometric analysis (particle size distribution, aspect ratio, circularity, porosity) does not require semantic class labels. It requires accurate object boundaries. Binary masks give you exactly that.

### 3.2 What Binary Buys Us Right Now

Binary segmentation unlocks every architecture candidate in our shortlist simultaneously:

**LBMS-SAM** (our primary track) was designed for dense, touching-particle segmentation in lithium battery SEM images. Its output is already binary — foreground particle versus background. The adapter modules (GSEFE, MDFF) it adds over frozen SAM are trained to refine boundaries, not to classify objects. Binary is its native output.

**RF-DETR** is an instance detection and segmentation model. With a single-class setup (class 0 = "object"), it produces per-instance binary masks with bounding boxes. Single-class RF-DETR is well-supported, fast to train, and benefits from NFFA-Europe's image-level category labels as weak supervision for what regions are likely to contain foreground objects.

**YOLOv11-seg** operates identically. Setting num_classes=1 with label "object" gives you per-instance binary masks at YOLO speed. This is a completely standard single-class instance segmentation setup with extensive real-world precedent.

**Attention U-Net** (already in the Phase 1 codebase) with a sigmoid output head and binary cross-entropy + Dice loss gives you a semantic binary map — not per-instance, but sufficient for porosity analysis and useful as a fast baseline.

Every one of these models becomes trainable the moment we have binary mask annotations. None of them require a class taxonomy.

### 3.3 The Upgrade Path

Binary is Phase 2A. Multi-class semantic segmentation — where we define material-specific classes and train per-class decoders — is Phase 2B, and it begins when two things are true: (1) Bharat Atomic has its own SEM hardware producing real images of specific customer materials, and (2) we have enough annotated data per material class to justify the class-specific annotation overhead. At that point, the binary masks from Phase 2A become the bootstrapping data for Phase 2B — re-annotators correct and extend binary masks into multi-class ones, rather than starting from scratch.

---

## 4. FiLM-Based Metadata Conditioning

Even within binary segmentation, not all SEM images are the same. A binary mask for particles in a powder sample requires the model to look for small, discrete, roughly circular foreground objects. A binary mask for MEMS structures requires finding large, rectangular, precisely-edged foreground regions. A binary mask for fibres requires finding elongated, curved, potentially overlapping foreground strands. These are the same binary task but with radically different spatial priors, and a single unconditional model has to learn all of them simultaneously from a potentially imbalanced training set.

Metadata conditioning — telling the model which material class it is looking at before it processes the image — allows the model to activate different internal feature channels for different material types, reducing the within-model interference between very different visual domains.

### 4.1 The OHE Scaling Problem and Why We Don't Use It

The naive implementation of metadata conditioning is One-Hot Encoding (OHE): a 10-dimensional binary vector where position i is 1 if the material is class i. The problem is that every time Bharat Atomic adds a new supported material type in a software update, the OHE vector grows by one dimension. The FiLM conditioning MLP now has a different input size. Its weights must be reinitialized. The network must be partially retrained for every material addition. For a shipping product, this is operationally broken — you cannot require a network surgery and retraining cycle every time you support a new customer material.

### 4.2 Learned Embedding Table — The Solution

The fix is straightforward: replace OHE with a learned embedding table. Every material class is assigned a row in a lookup table, and each row is a 64-dimensional dense vector of real numbers learned during training. The FiLM conditioning MLP always takes a 64-dimensional vector as input, regardless of how many material classes exist. Its weights never change shape.

When Bharat Atomic adds a new material type in a software update, the process is: add one new row to the embedding table, fine-tune that one row on a small set of images of the new material (a few hours of training), and ship the updated table as a lightweight file (~4KB for a 64-dim embedding). The rest of the network — the segmentation backbone, the GSEFE/MDFF adapters, the FiLM MLP — stays frozen. No architectural surgery. No full retraining.

This approach is well-validated in NLP (word embeddings, entity embeddings) and is directly applicable here. Embedding lookup is computationally free at inference time — it is a single table index operation before the first convolution.

### 4.3 The 10 NFFA Material Classes as Initial Conditioning Labels

The initial embedding table has 10 rows, one per NFFA-Europe category:

| Index | Class | Description | Typical SEM Appearance |
|---|---|---|---|
| 0 | Tips | AFM/SEM probe tips | Single sharp conical structure, high contrast |
| 1 | Particles | Discrete nano/micro particles | Small, roughly circular, often many per frame |
| 2 | Patterned Surfaces | Lithographically structured surfaces | Repeating geometric patterns, high regularity |
| 3 | MEMS Devices | Micro-electromechanical systems | Large rectangular features, precise edges |
| 4 | Nanowires | 1D elongated nanostructures | Thin, long, often tangled/crossed |
| 5 | Porous Sponge | Porous/foam-like materials | High void fraction, irregular interconnected pores |
| 6 | Biological | Cells, bacteria, tissue | Soft boundaries, varied morphology |
| 7 | Powder | Powder/aggregate samples | Irregular, often agglomerated, high particle density |
| 8 | Films / Coated Surfaces | 2D films and layered surfaces | Mostly uniform with visible grain or coating texture |
| 9 | Fibres | 1D fibre structures | Elongated, cylindrical, often overlapping |

The user selects their material from a dropdown in the SEM software UI. The selected class index is looked up in the embedding table, and the resulting 64-dim vector is passed into the FiLM conditioning layers. As Bharat Atomic collects real SEM images from its own hardware and gains more data on specific material sub-types, new entries are added to the table incrementally.

### 4.4 Auto-Classification Safety Net

The user-provided material label drives the conditioning, which means a wrong label actively hurts segmentation quality — the model will apply the wrong spatial priors. To catch obvious mistakes, a lightweight auto-classifier runs first:

```
Raw SEM image → DINOv2 features (frozen) → linear probe (10-class) → proposed label + confidence
```

This linear probe is a 2-hour training job on the NFFA-Europe classification task and gives strong accuracy. At inference time, the flow is:

```
User opens sample → SEM scans → auto-classifier proposes "Detected: Particles (89% confidence)" →
User confirms or overrides in UI → confirmed label drives FiLM conditioning → segmentation runs
```

The user is not forced to use the auto-classifier's suggestion — they can override it. The safety net exists to catch the common case where a user forgets to update the dropdown between samples. A caution warning is also shown in the UI when the user's selected label and the auto-classifier's suggestion disagree strongly (confidence gap above a threshold).

---

## 5. Model Architecture Candidates

Phase 2 runs multiple architectures in parallel. They address different points on the speed-quality-data-requirement tradeoff space, and the final production model selection will be made empirically after benchmarking on our own annotated data. Binary segmentation makes this multi-model strategy feasible because the annotation work is shared — one set of binary masks trains all of them.

### 5.1 Primary Track — LBMS-SAM (Frozen SAM + GSEFE + MDFF)

**What it is:** LBMS-SAM (Lithium Battery Material Segmentation — SAM) is a 2026 architecture from *Neural Networks* (Qi et al.) that keeps SAM's 312M-parameter ViT encoder and mask decoder entirely frozen and adds two small trainable adapter modules on top — a Sobel/Gabor Edge Feature Extractor (GSEFE) and a Wavelet-Denoised Multi-layer Feature Fusion module (MDFF) — totaling approximately 1.3M additional learnable parameters.

**Why it is the primary track:**

The open-set constraint (Section 2) means the backbone must retain SAM's general "find any object boundary" prior, which rules out full fine-tuning. But vanilla frozen SAM has a documented and consistent failure mode on dense SEM images: when particles or grains are touching or adhering to each other, SAM merges them into a single mask rather than separating them as individual instances. GSEFE and MDFF were specifically designed to fix this without touching SAM's weights.

GSEFE applies Sobel and Gabor operators (mathematical edge-detection transforms, not learned from data) to extract a dedicated edge/texture feature map, then passes it through a small enhancement network. This produces a boundary-emphasis signal that SAM's general-purpose encoder underweights, since SAM was trained on natural images where object boundaries are typically separated by clear spatial gaps rather than contact zones.

MDFF applies a discrete wavelet transform to SAM's intermediate feature maps, soft-thresholds the result (denoising in the wavelet domain, analogous to what we did in Phase 1 for the images themselves), inverts the transform, and fuses the result back into the feature stream with a small convolutional block. This suppresses high-frequency noise that the Phase 1 denoiser may not fully eliminate from SAM's intermediate activations, particularly on fine-grained textures like grain boundaries in ceramic or metal alloy images.

**Results from the original paper (lithium battery SEM):** LBMS-SAM achieved IoU of 96.2% and BIoU of 88.4% on its primary test set, outperforming vanilla SAM (IoU 84.5%), HQ-SAM (IoU 86.1%), and µSAM (IoU 92.4%) on dense adherent particle segmentation. All 312M parameters of SAM were frozen throughout.

**What this means for our setup:** the 1.3M trainable parameters of GSEFE + MDFF train quickly even on our small initial dataset. SAM's frozen prior gives us a working zero-shot baseline from day one, before any training begins. GSEFE and MDFF are ablated separately before training jointly — see Section 8.

**Important note on µSAM vs. LBMS-SAM:** µSAM ("Segment Anything for Microscopy," Archit et al., *Nature Methods* 2025) is a fine-tuned SAM variant specifically for microscopy and is a natural candidate to reach for. However, µSAM's mask decoder uses a watershed-based approach that relies on global spatial context, and this is documented to underperform specifically on high-density, touching-particle segmentation — which is our primary failure mode. LBMS-SAM's adapter approach consistently outperforms µSAM on this failure mode. We use µSAM as a benchmark comparison, not as a starting checkpoint.

### 5.2 Secondary Track — RF-DETR

**What it is:** RF-DETR (Real-Time Detection Transformer) is a 2025 single-stage instance detection and segmentation model that achieves near-SOTA accuracy at real-time speeds, built on a DETR-style transformer architecture with optimizations for deployment. It produces per-instance bounding boxes and segmentation masks in a single forward pass.

**Why it is in the running:** for material categories where particles or objects are well-separated (not touching), RF-DETR running in single-class mode (one class: "object") is fast, accurate, and has a mature ONNX/TensorRT export path. The DETR-style set prediction (no NMS needed) makes it cleaner than YOLO for dense, overlapping objects because it explicitly models object-object relationships.

**Its limitation:** RF-DETR is a closed-set model — if we train it on binary "object" masks across our 10 material classes, it learns a merged prior across all of them. Unlike LBMS-SAM, it has no zero-shot fallback for a completely new material type. It also does not naturally handle severe object adhesion, where the boundary between two touching particles is a hairline shared edge. For materials with those properties (powder, biological, some particles), LBMS-SAM will outperform it.

**Its role:** RF-DETR runs as a parallel candidate for speed benchmarking. If it meets the 40ms latency target on CUDA hardware while coming close to LBMS-SAM accuracy on our test set, it may become the preferred deployment model for specific material categories where its limitations don't apply.

### 5.3 Secondary Track — YOLOv11-seg

**What it is:** YOLOv11-seg (Ultralytics, 2024) is the current generation of the YOLO instance segmentation family. It uses a backbone + neck + detection head architecture with a lightweight mask generation module producing per-instance binary masks alongside bounding boxes in a single forward pass.

**Why it is in the running:** YOLO is the fastest credible instance segmentation model available, with the most mature TensorRT export tooling of any model in our shortlist. Real-world benchmarks show YOLOv8-seg derivatives reaching up to 374 FPS on an RTX 5070 Ti with TensorRT FP16 — orders of magnitude below our 40ms budget. For material categories with clean, well-separated foreground objects, YOLO's speed advantage is real and its accuracy at single-class binary segmentation is proven.

**Its limitations:** identical to RF-DETR — closed-set, no zero-shot fallback, struggles with touching objects, no open-set generalization. Additionally, YOLO's anchor-based detection mechanism and NMS post-processing are less suited than RF-DETR's set-prediction approach for high-density, overlapping particle fields.

**Its role:** YOLO runs as the speed ceiling benchmark. If YOLO meets the accuracy bar on our test set, it may become the deployed fast-path for specific known material verticals. If it doesn't, its failure modes inform where LBMS-SAM's open-set approach is genuinely necessary versus overkill.

**On PerovSegNet (the YOLO-based materials SEM precedent):** PerovSegNet (Pan et al., 2025) built an improved YOLOv8x on perovskite solar cell SEM images, achieving 87.25% mAP on three material-specific classes with 10,994 training images. It demonstrates that YOLO-family architectures absolutely can work on materials SEM segmentation — but it is a closed-set, single-material-system model that needed ~11,000 annotated images for 3 classes. Its architectural ideas (Adaptive Shuffle Dilated Convolution for multi-scale features, Separable Adaptive Downsampling for boundary preservation) are worth borrowing for our GSEFE/MDFF design, but its training data requirement rules it out as a primary path at our current annotation scale.

### 5.4 Baseline — Attention U-Net with FiLM Conditioning

**What it is:** the Attention U-Net already implemented in `semPhase1/notebooks/models/AttentionNet.py`, extended with FiLM conditioning layers in the decoder path. Sigmoid output, 1-channel binary mask.

**Why it is the baseline:** this is the architecture we understand best — it is already in the codebase, we understand its failure modes from Phase 1 denoising, and its encoder-decoder structure maps cleanly onto the binary foreground/background task. FiLM conditioning (material embedding → MLP → γ and β vectors → applied to each decoder block's feature maps) slots in with minimal code changes.

**What it gives us:** a material-conditioned binary semantic segmentation map. Not per-instance — it cannot separate touching objects into individual mask IDs — but for porosity/defect-area analysis where individual object identity doesn't matter, this is sufficient and fast. For particle counting and PSD where per-instance masks are needed, the semantic output of the Attention U-Net can be post-processed with connected component analysis, which is the standard approach and works when particles are mostly non-touching.

**Its role:** primary baseline for semantic binary segmentation, primary test bed for the FiLM conditioning implementation, and the fast-path for porosity analysis. All new conditioning code is prototyped here before being integrated into LBMS-SAM.

### 5.5 Lightweight Alternatives — MobileSAM / FastSAM

MobileSAM (distilled ViT-Tiny encoder, full SAM decoder) and FastSAM (YOLOv8-based architecture that reformulates segmentation as an everything-at-once instance task) are both designed to bring SAM-level open-set segmentation within a latency budget that full SAM ViT-L cannot meet. Neither has materials-SEM precedent yet, so both require validation on our own test set before being trusted. They are benchmarked alongside the primary and secondary tracks for latency vs. accuracy tradeoff analysis, not assumed to work.

### 5.6 Zero-Training Booster — SAM-I-Am Style Post-Processing

SAM-I-Am (Abebe et al., 2024) is a rule-based, training-free post-processing layer over vanilla SAM for atomic-scale electron micrographs. It extracts geometric and textural features from SAM's intermediate mask proposals to perform mask removal (removing spurious detections) and mask merging (correcting over-segmentation of single objects). It reported absolute IoU gains of +21.35%, +12.6%, and +5.27% over vanilla SAM across three difficulty tiers, with zero additional training.

This is worth running as a zero-cost baseline against frozen SAM before any training investment. If rule-based post-processing closes most of the accuracy gap, it changes the priority order of training the GSEFE/MDFF modules. If it doesn't, that comparison is the evidence that confirms GSEFE/MDFF are necessary. Either way, running this costs nothing except evaluation time.

### 5.7 Architecture Comparison Table

| Architecture | Open-Set? | Binary Native? | Training Data Needed | Expected Latency (CUDA TensorRT) | Primary Role |
|---|---|---|---|---|---|
| **Frozen SAM + GSEFE + MDFF (LBMS-SAM)** | ✅ Yes | ✅ Yes | ~200 images seed set | 30–60ms (ViT-L backbone) | Primary production candidate |
| **RF-DETR (single-class)** | ❌ No | ✅ Yes | ~200 images seed set | 15–30ms | Speed/quality benchmark |
| **YOLOv11-seg (single-class)** | ❌ No | ✅ Yes | ~200 images seed set | 5–15ms | Speed ceiling benchmark |
| **Attention U-Net + FiLM** | ❌ No | ✅ Yes | ~200 images seed set | 20–40ms | Semantic baseline, porosity |
| **MobileSAM / FastSAM** | ✅ Yes | ✅ Yes | ~200 images seed set | 10–25ms | Lightweight open-set candidate |
| **SAM-I-Am post-processing** | ✅ Yes | ✅ Yes | Zero — no training | Same as base SAM | Free zero-training baseline |

---

## 6. Dataset Strategy

### 6.1 What NFFA-Europe Gives Us and What It Doesn't

The NFFA-Europe Majority dataset (used in Phase 1) contains approximately 21,000–25,500 SEM images at 1,024×728 pixels, classified into 10 categories by nanoscientist consensus. It is a pure image-classification dataset. It has no segmentation masks, no instance annotations, and no bounding boxes. The 10 category labels are image-level — they say "this whole image is particles" but say nothing about which pixels are particle and which are background.

This means NFFA-Europe cannot be used as a direct supervised segmentation training source. It is useful in Phase 2 in two specific ways: as a large pool of unlabeled SEM images for potential self-supervised pretraining experiments (Section 8.3), and as a pool of category-diverse images to annotate using µSAM-assisted bootstrapping. The 10 class labels also serve directly as the initial FiLM conditioning labels (Section 4.3), so the image-level labels are not wasted — they condition the segmentation model even though they are not pixel-level.

One detail worth noting for the eventual Phase 2.1 calibration work: NFFA-Europe images include a white instrument label bar at the bottom of each frame showing scale and acquisition settings. This bar is cropped out in all Phase 1 and Phase 2 training pipelines, but for the scale calibration step in morphometric analysis, an equivalent bar exists on Bharat Atomic's own SEM output and must be parsed rather than discarded.

### 6.2 Our Real Starting Point — Zero Labeled Masks

We have no segmentation masks. The realistic near-term annotation target is approximately 200 source images with SAM-assisted binary mask correction, not thousands. This is not a disaster — it is the reason the architecture choices in Section 5 were made the way they were:

- Frozen SAM gives a working zero-shot segmentation from before any training starts
- GSEFE + MDFF require only ~1.3M parameters to fine-tune, not 300M
- Binary masks require less annotation expertise and time than multi-class masks
- The Phase 1 noise simulator provides free, realistic augmentation once any masks exist

The unit that matters is **masks, not images**. 200 source images of a particle-dense powder sample might yield 5,000 individual particle masks if particles are small. 200 images of MEMS structures might yield 400 masks if structures are large. This should be measured empirically on a sample of our own images before making assumptions about data scarcity.

### 6.3 Annotation Pipeline — µSAM-Assisted Bootstrapping

The annotation workflow is designed to minimize manual pixel-drawing. The process:

1. Run frozen SAM in automatic mode (zero-shot, no training, no prompts) on raw SEM images
2. Visually inspect the SAM output to assess: how many objects are correctly separated? How many touching objects are merged? How many spurious detections are there?
3. Accept correct masks, correct merged masks by clicking point prompts to separate instances, delete spurious detections
4. Export corrected masks in COCO JSON format with the image-level material class as metadata

This is dramatically faster than drawing masks from scratch. Annotators are correcting SAM's output, not creating from nothing. For well-separated objects like isolated particles, frozen SAM's proposals are often accepted with minimal correction. For touching particles or dense powder aggregates, more correction is needed — which is exactly the failure mode GSEFE/MDFF are trained to fix, so these corrected examples are the most valuable training data.

**Annotation tooling:** Label Studio with a SAM ML backend provides this workflow in an open-source, self-hostable setup. µSAM's napari plugin is a good alternative for researchers more comfortable with image analysis tools. Both support point-prompt correction of SAM proposals, which is the critical capability.

**What we are not using:** vanilla LabelMe for polygon drawing from scratch. It costs 10x the annotation time and produces no additional quality benefit over SAM-corrected masks.

### 6.4 Phase 1 Noise Simulator as a Free Augmentation Engine

This is a genuine Phase 2 advantage that most precedent SEM segmentation papers did not have. Once we have binary masks annotated on clean (or denoised) images, the Phase 1 `NoiseImage`/`new_augment_sem` engine can generate arbitrarily many realistic noisy variants of the same image-mask pair:

```python
# Same mask, different noise realizations — free training data
for _ in range(10):
    noisy_image, metadata = new_augment_sem(clean_crop)
    training_pairs.append((noisy_image, binary_mask))
```

This is strictly better than standard geometric augmentation (flips, rotations, crops) because it changes the noise statistics of the input while keeping the mask correct. The model sees the same object boundaries under different noise regimes, which directly improves robustness to real SEM imaging conditions that vary with accelerating voltage, beam current, and sample conductivity. Standard geometric augmentation only changes the object's position and orientation — the noise regime stays the same.

### 6.5 Annotation Scale Targets

| Stage | Target | Rationale |
|---|---|---|
| Unlabeled pool (self-supervised experiments only) | Full NFFA-Europe, ~21,000–25,500 images | Already available from Phase 1 |
| Seed annotated set — initial training target | ~200 source images, mask count TBD | Realistic near-term capacity; count instances per image first |
| LBMS-SAM reference scale (aspirational) | 244 images / ~13,000 masks | LBMS-SAM paper's training set |
| Synthetic augmentation via Phase 1 noise engine | As large as needed, no annotation cost | Free |
| Pseudo-labeled expansion | Spot-checked, not fully re-annotated | Only once a working seed model exists and is validated |

---

## 7. Loss Function Design for Binary Segmentation

Standard binary cross-entropy loss fails on SEM segmentation because foreground objects are typically a small fraction of total pixels — in a particle image, the particles themselves might cover 20–30% of the image area, and pores in a porous sponge image might cover 5–10%. Pure BCE on imbalanced pixel distributions causes the model to learn to predict "everything is background" and achieve high accuracy while being useless.

The solution is a composite loss that explicitly penalizes foreground miss-detections:

```
Loss = α × BCE(pred, target) + β × (1 − Dice(pred, target)) + γ × FocalLoss(pred, target)
```

**BCE** provides per-pixel probability calibration. **Dice loss** directly optimizes the intersection-over-union between predicted and true foreground regions, making it invariant to class imbalance by construction. **Focal loss** down-weights the contribution of easy, correctly-classified background pixels and focuses gradient signal on hard boundary pixels and misclassified foreground pixels. The combination of all three is standard practice in medical image segmentation and transfers directly to SEM.

Additionally, **boundary-weighted loss** adds extra penalty on pixels near annotated object boundaries — the most informative and hardest-to-get-right region of any mask. This directly improves the Boundary IoU (BIoU) metric, which measures mask quality specifically at boundaries and is more meaningful than standard IoU for thin, touching, or irregular objects.

Coefficients α, β, γ are tuned per architecture on the validation set.

---

## 8. Training Strategy

### 8.1 Freeze Strategy

The Phase 1 lesson — freeze the broadly useful part, fine-tune only the domain-specific part — applies identically here.

For **LBMS-SAM:** SAM's ViT encoder and mask decoder are frozen entirely. Only GSEFE and MDFF are trained. SAM has never seen SEM images and its pretrained SA-1B prior is exactly the general "find object boundaries" capability we want to preserve. Full fine-tuning of SAM on our small labeled set would cause catastrophic forgetting — the original LBMS-SAM paper demonstrated this directly, showing IoU drop from 95.6 to 80.5 and BIoU drop from 86.5 to 52.6 when all parameters were unfrozen on their larger dataset. With our smaller dataset, the collapse would be worse.

For **RF-DETR and YOLO:** fine-tune the full model. Both are designed for full fine-tuning on custom datasets and do not have the same "pretrained general prior" vulnerability as SAM, because their ImageNet or COCO-pretrained weights are being adapted to a new task (SEM single-class instance segmentation) rather than a narrow domain of an already-learned task.

For **Attention U-Net:** train from scratch on the binary SEM task, consistent with Phase 1. This is the model where FiLM conditioning is prototyped and debugged before being integrated into more complex architectures.

### 8.2 Training Order — Do This, Not That

The order of operations matters more than most planning documents admit. The following sequence minimizes wasted compute and ensures every training decision is based on empirical evidence from the previous step:

**Step 1:** Run frozen SAM (zero-shot, automatic mode, no training) on 20–30 raw SEM images. Visually assess output. This is free and tells you the baseline quality and the specific failure modes that training needs to fix.

**Step 2:** Run SAM-I-Am-style rule-based post-processing on the Step 1 output. If this closes most of the accuracy gap against hand-corrected masks, GSEFE/MDFF training is less urgent.

**Step 3:** Annotate the seed set (~200 images) using µSAM-assisted bootstrapping. Augment with Phase 1 noise engine.

**Step 4:** Train GSEFE alone (MDFF frozen/absent). Evaluate on held-out set. Record IoU, BIoU, Dice, boundary accuracy.

**Step 5:** Train MDFF alone (GSEFE absent). Evaluate on held-out set. Compare against Step 4.

**Step 6:** Train GSEFE + MDFF jointly. Compare against Steps 4 and 5. The combination should improve over either alone but the margin tells you which module is doing more work and where to focus future improvement effort.

**Step 7 (parallel):** Train Attention U-Net with FiLM conditioning on the same seed set. This gives a fast, lightweight alternative and validates the FiLM conditioning implementation.

**Step 8 (parallel):** Fine-tune RF-DETR and YOLOv11-seg on the same seed set in single-class mode. Benchmark latency vs. accuracy against primary track.

**Step 9:** Full model comparison on the held-out evaluation set. Select production model based on accuracy × latency × open-set requirement.

### 8.3 Self-Supervised Pretraining — What's Worth Doing and What Isn't

There is a tempting but partially wrong instinct here: "we have tens of thousands of unlabeled NFFA-Europe images, so let's pretrain everything self-supervised first." What's right about this: masked autoencoder SSL (MAE-style) on SEM images does improve downstream segmentation accuracy at small annotated data scales, as demonstrated by the npj Computational Materials ConvNeXtV2/FCMAE paper (Rettenberger et al., 2025), which showed SEM-domain SSL beating ImageNet-pretrained weights using as few as ~1,000 unlabeled pretraining images.

What's wrong: **GSEFE and MDFF cannot meaningfully benefit from SSL.** GSEFE applies fixed mathematical transforms (Sobel, Gabor operators) and a small enhancement network. MDFF applies a fixed wavelet transform and a small fusion convolution. Together they have ~1.3M parameters — there is no large representation-learning problem here for a self-supervised pretext task to do useful work on. These modules go straight to supervised training.

**Full fine-tuning of SAM's ViT encoder with SSL** on NFFA-Europe is a legitimate but genuinely uncertain experiment. SAM's encoder is a ViT-L and standard ViT-SSL recipes assume far more data (100M+ images) than NFFA-Europe provides. If this experiment is run, it must be gated: compare zero-shot mask quality of the SSL-adapted encoder versus vanilla frozen SAM on a held-out sample before it touches any labeled data. If the adapted encoder is not clearly better, discard the experiment and retain vanilla SAM. This is scheduled after the primary GSEFE/MDFF track has a working baseline.

---

## 9. Evaluation Metrics

Three metrics are tracked for all model comparisons:

| Metric | What It Measures | Why It Matters for SEM |
|---|---|---|
| **IoU (Intersection over Union)** | Overlap between predicted and true mask across all pixels | Standard segmentation benchmark; catches both false positives and false negatives |
| **BIoU (Boundary IoU)** | IoU computed only within a narrow band around object boundaries | SEM segmentation quality lives at the boundary — two models can have similar IoU but wildly different boundary quality, and downstream measurement accuracy depends directly on boundary accuracy |
| **Dice Coefficient** | 2×TP / (2×TP + FP + FN) | Standard for imbalanced segmentation; equivalent to F1-score on pixel predictions |
| **Fa (F-score at boundary)** | F-score specifically on foreground boundary pixels | Complementary to BIoU for evaluating thin, precise boundaries like grain edges |
| **Inference Latency (ms)** | End-to-end wall-clock time per image on target hardware | Production requirement; 40ms hard ceiling on CUDA + TensorRT |
| **Instance Recall / Precision** | Fraction of true instances correctly detected / fraction of detections that are correct | For instance segmentation specifically — checks whether all individual objects are found, not just whether pixels are correctly classified |

NFFA-Europe's 10 class labels also enable a classification accuracy metric on the auto-classifier (DINOv2 linear probe) as a standalone evaluation, separate from segmentation quality.

---

## 10. Latency Budget and Deployment Reality

The hard latency ceiling is 40ms per frame. This must be evaluated on the actual target hardware — CUDA + TensorRT on the NVIDIA GPU that is incoming for Phase 1 GPU-scale training — not on the current M3 Max MPS setup, where CUDA-optimized kernels are unavailable and TensorRT does not run.

The 40ms budget must cover the full inference pipeline, not just the model forward pass:

```
Image preprocessing (resize, normalize, channel conversion) — ~1–3ms
Denoiser forward pass (Phase 1 NAFNet) — ~15–25ms at 1K on RTX 4090 + TensorRT (projected)
Segmentation forward pass — target: 15–25ms remaining budget
Mask postprocessing (upsampling, NMS or merge logic) — ~2–5ms
```

This means the segmentation model has roughly 15–20ms of the total budget if the Phase 1 denoiser runs in the same pipeline loop. The realistic options for hitting this:

- YOLOv11-seg / RF-DETR: comfortably within budget on CUDA hardware
- MobileSAM/FastSAM: likely within budget, needs validation
- LBMS-SAM (ViT-L backbone): likely above budget at 1K resolution; a ViT-B backbone variant should be evaluated
- Full SAM ViT-L: definitely above budget without architectural changes

The morphometric analysis pipeline (Phase 2.1) is explicitly decoupled from the real-time segmentation loop. Segmentation runs in real time. Measurements (PSD, aspect ratio, circularity) are computed asynchronously on a slower cadence, triggered per-frame but completing in a separate thread. There is no published precedent for running full PSD analysis within a 40ms real-time budget, and there is no user requirement for it either — users want to see clean segmented images in real time, not updated measurement histograms at 25 fps.

---

## 11. Phase 2.1 — Morphometric Analysis Pipeline

Once stable segmentation masks are available, a downstream measurement pipeline computes physical quantities from each mask. This is not a novel research problem — the computational methods are well-established. The engineering work is in connecting the segmentation output to calibrated physical measurements.

**Scale calibration:** the SEM's own scale bar and acquisition metadata contain the information needed to convert pixel dimensions to physical units (nm, µm). The calibration process measures the pixel span corresponding to a known physical length on the scale bar, yielding a pixel-to-length ratio. This must run automatically from the SEM's output metadata when available, with manual entry as a fallback.

**Per-instance measurement:** for each binary instance mask, standard connected-component analysis computes:
- Area (in calibrated physical units²)
- Equivalent circular diameter (diameter of a circle with the same area)
- Aspect ratio (major axis / minor axis from the mask's geometric moments)
- Circularity (4π × area / perimeter²; perfect circle = 1.0)
- Convexity (mask area / convex hull area; 1.0 = fully convex)

These are derived from the mask geometry alone — no additional model inference needed.

**Particle size distribution (PSD):** histogram of equivalent circular diameters across all detected instances in the image or image stack, expressed in physical units (nm or µm). Standard output format for materials characterization.

**Porosity analysis:** total pore area as a fraction of total image area (from the semantic binary map — pores are background in a porous material image). Bin-wise size distribution of individual pore instances.

**3D / volume estimates:** true volumetric analysis requires actual 3D acquisition (FIB-SEM serial sectioning), which is a separate capability. For Phase 2.1, volume is a stereological estimate — equivalent spherical diameter cubed, multiplied by π/6 — which is a standard approximation for quasi-spherical particles. Non-spherical particles require explicit shape-factor correction, which is out of scope for Phase 2.1 but can be added once aspect ratio and convexity data are available.

---

## 12. Repository Structure (Planned)

```
bharatAtomic/
│
├── README.md                            ← Parent-level project overview (Phase 1–3)
│
├── semPhase1/                           ← Phase 1: Denoising pipeline (complete)
│   ├── README.md
│   ├── dataset/crop/images/             ← NFFA-Europe SEM dataset (256×256 crops)
│   └── notebooks/
│       ├── base.py                      ← Shared pipeline: data, noise, training, eval
│       ├── models/
│       │   ├── rdunet.py
│       │   ├── NAFNet_arch.py
│       │   ├── AttentionNet.py
│       │   └── dncnn.py
│       └── *.ipynb                      ← Per-model training notebooks
│
├── semPhase2/                           ← Phase 2: Segmentation pipeline (this phase)
│   ├── README.md                        ← This file
│   ├── dataset/
│   │   ├── raw/                         ← NFFA-Europe source images (from Phase 1)
│   │   ├── annotated/                   ← Binary mask annotations (COCO JSON format)
│   │   └── augmented/                   ← Noise-augmented image-mask pairs
│   └── notebooks/
│       ├── base_seg.py                  ← Shared pipeline: data loading, metrics, eval
│       ├── models/
│       │   ├── lbms_sam/
│       │   │   ├── gsefe.py             ← Sobel/Gabor Edge Feature Extractor
│       │   │   ├── mdff.py              ← Wavelet Denoised Multi-layer Feature Fusion
│       │   │   └── lbms_sam.py          ← Full LBMS-SAM wrapper over frozen SAM
│       │   ├── attention_unet_film.py   ← Attention U-Net + FiLM conditioning
│       │   ├── embedding_table.py       ← 10-class learned embedding + FiLM MLP
│       │   └── auto_classifier.py       ← DINOv2 linear probe for safety net
│       ├── annotation/
│       │   ├── bootstrap_sam.py         ← Run zero-shot SAM on raw images for annotation seed
│       │   └── augment_masks.py         ← Apply Phase 1 noise engine to annotated pairs
│       ├── lbms_sam.ipynb               ← LBMS-SAM training and evaluation
│       ├── rfdetr.ipynb                 ← RF-DETR single-class training and evaluation
│       ├── yolo_seg.ipynb               ← YOLOv11-seg single-class training and evaluation
│       └── attention_unet_film.ipynb    ← Attention U-Net + FiLM baseline
│
├── semPhase3/                           ← Phase 3: RL calibration (planned)
└── semPhase3/README.md
```

---

## 13. Key Design Decisions

| Decision | Rationale | Outcome |
|---|---|---|
| Binary segmentation in Phase 2A | Eliminates annotation taxonomy overhead; makes all model candidates trainable from the same mask set; sufficient for all downstream morphometric measurements | All architectures trainable on a single shared annotation effort |
| Open-set primary track (LBMS-SAM) | Product requirement: any material must be segmentable; closed-set detectors fail on unseen materials | Graceful degradation on new materials; zero-shot fallback from day one |
| Closed-set models (RF-DETR, YOLO) as secondary benchmarks | Speed ceiling and accuracy ceiling reference; valid deployment path for specific known material verticals | Empirical comparison determines which model ships for which use case |
| Learned embedding table over OHE | OHE requires network surgery for every new material type; embedding table adds new materials as a new row, no architecture change | Scaling to new material types costs hours of fine-tuning, not weeks of retraining |
| Freeze SAM encoder + decoder entirely | LBMS-SAM ablation: full fine-tuning on domain data collapsed IoU from 95.6 to 80.5; frozen prior is the valuable asset | SAM's general object-boundary prior is preserved; only domain-specific adapters are trained |
| Annotate on denoised images | Phase 1 denoiser output is cleaner; annotators produce higher quality masks on clean images; model receives consistent input at inference | Annotation quality and model consistency improved at no extra cost |
| Phase 1 noise engine as augmentation | Realistic, material-aware noise augmentation vs. generic geometric transforms | Free, domain-specific training data expansion without additional annotation |
| Decouple morphometrics from real-time loop | No published precedent for 40ms PSD/circularity computation; not a user-visible requirement in real time | 40ms segmentation target is achievable; measurement runs asynchronously |
| FiLM conditioning in decoder path only | Encoder features are spatially shared and don't need per-material specialization; decoder is where object-class-specific reconstruction happens | Minimal parameter overhead; conditioning influences final mask generation, not early feature extraction |

---

## 14. Checklist

**Foundational — before any training**
- [ ] Run frozen SAM (zero-shot, automatic mode) on 20–30 real Bharat Atomic SEM images; visually assess merge/split failure rate per material class
- [ ] Run SAM-I-Am-style rule-based post-processing on zero-shot SAM output; compare against hand-corrected masks to establish zero-training baseline
- [ ] Set up Label Studio with SAM ML backend for annotation
- [ ] Count actual mask instances per image on a representative sample — establish the real annotation unit before assuming data scarcity
- [ ] Build held-out evaluation split (hand-verified) before any training begins; this set must never be touched during training decisions

**Annotation**
- [ ] Annotate seed set (~200 source images via SAM-assisted correction)
- [ ] Apply Phase 1 `NoiseImage`/`new_augment_sem` to annotated pairs for augmentation
- [ ] Store all annotations in COCO JSON format with image-level material class metadata attached

**Primary track — LBMS-SAM**
- [ ] Implement GSEFE (Sobel/Gabor edge extractor + small enhancement network)
- [ ] Implement MDFF (wavelet denoising + multi-layer fusion convolution)
- [ ] Train GSEFE alone; evaluate on held-out set vs. frozen SAM baseline
- [ ] Train MDFF alone; evaluate on held-out set vs. frozen SAM baseline
- [ ] Train GSEFE + MDFF jointly; evaluate and compare against both isolated results
- [ ] Implement FiLM conditioning on LBMS-SAM decoder path using 10-class embedding table

**Baseline and secondary tracks**
- [ ] Implement FiLM conditioning on Attention U-Net (from Phase 1 codebase)
- [ ] Train Attention U-Net + FiLM on seed set; evaluate binary semantic segmentation quality
- [ ] Fine-tune RF-DETR in single-class mode on seed set
- [ ] Fine-tune YOLOv11-seg in single-class mode on seed set
- [ ] Train DINOv2 linear probe auto-classifier on NFFA-Europe 10-class labels

**Latency and deployment**
- [ ] Benchmark all models on CUDA hardware (incoming) with TensorRT FP16
- [ ] Benchmark MobileSAM/FastSAM as lightweight open-set alternatives
- [ ] Confirm morphometric pipeline is decoupled from the real-time segmentation loop
- [ ] Validate 40ms end-to-end budget: denoiser + segmentation + postprocessing

**Deferred**
- [ ] Self-supervised continued pretraining of SAM encoder on NFFA-Europe (only after primary track has a working baseline; must pass zero-shot validation gate before touching labeled data)
- [ ] Multi-class semantic segmentation (Phase 2B, after Bharat Atomic SEM hardware is available)
- [ ] Closed-set vertical-specific fast-path (Phase 2B, after defined customer vertical with annotation budget)
- [ ] Phase 2.1 morphometric pipeline (after segmentation output format is stable)

---

## 15. Remaining Work and Roadmap

### Phase 2A (Current — Binary Open-Set Segmentation)
- [ ] Full training runs on NVIDIA hardware with all architecture candidates
- [ ] Real SEM image validation — inference on actual Bharat Atomic hardware images
- [ ] Per-model comparison table: IoU, BIoU, Dice, Fa, latency on identical hardware
- [ ] Production inference script — standalone, no notebook dependency, takes image path + material label as inputs

### Phase 2B *(Planned — after hardware)*
- [ ] Multi-class semantic segmentation with material-specific class taxonomies
- [ ] Closed-set fast-path detector for high-volume customer material verticals
- [ ] Phase 2.1 scale calibration and morphometric measurement pipeline
- [ ] Multimodal SE + BSE + EDS (X-Ray) input fusion

### Phase 3 *(Planned)*
- [ ] RL formulation for autonomous SEM parameter calibration
- [ ] State encoder from Phase 1 denoiser + Phase 2 segmenter as observation input to RL agent
- [ ] Simulation environment for RL policy training before hardware deployment
- [ ] Closed-loop testing on live SEM system

---

## 16. References

1. Qi, Y., Zhang, J., Kuang, J., Ren, T., Wang, D., Wu, Z., Zheng, H., & Zhang, Q. (2026). LBMS-SAM: Segment anything model guided SEM image segmentation for lithium battery materials. *Neural Networks*, 196, 108325.
2. Archit, A., et al. (2025). Segment Anything for Microscopy. *Nature Methods*, 22, 579–591.
3. Pan, J. G., Wang, L., & Cai, X. (2025). Automated and Scalable SEM Image Analysis of Perovskite Solar Cell Materials via a Deep Segmentation Framework (PerovSegNet). arXiv:2509.26548.
4. Rettenberger, L., Szymanski, N. J., Giunto, A., Dartsi, O., Jain, A., Ceder, G., Hagenmeyer, V., & Reischl, M. (2025). Leveraging unlabeled SEM datasets with self-supervised learning for enhanced particle segmentation. *npj Computational Materials*, 11, 289.
5. Abebe, W., et al. (2024). SAM-I-Am: Semantic boosting for zero-shot atomic-scale electron micrograph segmentation. arXiv:2404.06638.
6. Wang, A., et al. (2025). SAM-EM: Real-Time Segmentation for Automated Liquid Phase Transmission Electron Microscopy. arXiv:2501.03153.
7. Kirillov, A., et al. (2023). Segment Anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 4015–4026.
8. Oquab, M., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. *Transactions on Machine Learning Research*.
9. Rettenberger, L., et al. (2024). Uncertainty-aware particle segmentation for electron microscopy at varied length scales. *npj Computational Materials*, 10, 124.
10. Rühle, B., Krumrey, J. F., & Hodoroaba, V.-D. (2021). Workflow towards automated segmentation of agglomerated, non-spherical particles from electron microscopy images using artificial neural networks. *Scientific Reports*, 11, 4942.
11. Machine vision-driven automatic recognition of particle size and morphology in SEM images. *Nanoscale* (RSC Publishing), 2020. DOI:10.1039/D0NR04140H.
12. Woo, S., et al. (2023). ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. *CVPR 2023*, 16133–16142.
13. Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI 2018*.
14. Oktay, O., et al. (2018). Attention U-Net: Learning where to look for the pancreas. *MIDL 2018*. arXiv:1804.03999.
15. Aversa, R., et al. (2018). The first annotated set of scanning electron microscopy images for nanoscience (NFFA-Europe). *Scientific Data*, 5, 180172.
16. Hirabayashi, Y., et al. (2024). Deep learning for three-dimensional segmentation of electron microscopy images of complex ceramic materials. *npj Computational Materials*, 10, 46.
17. Foundation Models for Zero-Shot Segmentation of Scientific Images without AI-Ready Data. arXiv:2506.24039, 2025.
