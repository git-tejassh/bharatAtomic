# Phase 2 — Real-Time SEM Segmentation & Quantitative Microstructure Analysis: Research Foundation & Strategy

> Companion planning document to `README.md` (Phase 1) and `semPhase1/README.md`. This covers the literature review, methodology, model selection, and dataset strategy for Phase 2: real-time defect/particle segmentation, with morphometric analysis (PSD, aspect ratio, circularity, porosity) planned as a downstream consumer of the segmentation output.

---

## 1. Scope and Constraints, Restated

Before picking models, it's worth pinning down what "real-time" and "segmentation" actually mean for this project, because both words cover a wide range of very different engineering problems:

- **Latency target:** under 40 ms per frame, hard ceiling, no exceptions. This is a per-frame inference budget, not an average — a model that's 25 ms most of the time but spikes to 80 ms on dense particle fields fails the requirement.
- **Task type:** instance segmentation (need per-object masks to compute per-particle area, aspect ratio, circularity — not just semantic class maps), though semantic segmentation is the right tool for porosity/defect-area tasks where individual object identity doesn't matter.
- **Deployment hardware:** unspecified in the Phase 1 docs beyond the M3 Max dev machine and the "incoming NVIDIA GPU" mentioned for Phase 1. The 40 ms target effectively assumes a CUDA + TensorRT path — this is addressed in Section 4.
- **Downstream consumer:** the segmentation masks feed a measurement pipeline (PSD, surface area, volume estimates, aspect ratio, circularity, porosity), which depends on (a) accurate masks and (b) a pixel-to-physical-unit calibration step using the SEM's scale bar/metadata. This second part has no real precedent for real-time operation in the literature — it's traditionally a batch, offline analysis step (see Section 6).

---

## 2. Literature Review

### 2.1 Deep learning for EM/SEM segmentation — surveys and reviews

These give the lay of the land for how DL segmentation has been applied to electron microscopy broadly, mostly in connectomics/cellular EM rather than materials SEM, but the architectural lessons transfer:

- **Aswath, A. et al., "Segmentation in large-scale cellular electron microscopy with deep learning: A literature survey"** (arXiv:2206.07171, 2022). Surveys how semantic and instance segmentation architectures were adapted for cellular/sub-cellular EM structures, examining the special challenges posed by EM images and the network architectures that addressed them, alongside a review of the major datasets that enabled progress. Good starting point for understanding why generic Cityscapes/COCO-tuned architectures don't transfer cleanly to microscopy.
- **Khadangi, A., Boudier, T., & Rajagopal, V., "EM-stellar: benchmarking deep learning for electron microscopy image segmentation"** (*Bioinformatics*, 37(1), 2021). Notes that the inherent low contrast of EM datasets is a major challenge for rapid segmentation of cellular ultrastructures, especially with high-resolution datasets from electron tomography and serial block-face imaging — and that no rigorous benchmark of DL methods existed prior to this work. Directly relevant to your noise/contrast situation, since you already characterized SEM-specific noise sources in Phase 1.
- **"Deep learning for brain electron microscopy segmentation"** (*Computers & Graphics*, Sept 2025). A 2025 meta-analysis reviewing 60 deep learning approaches for brain EM segmentation, covering self-supervised learning to reduce manual annotation needs, topology-aware loss functions for better continuity in neuron segmentation, and transformer architectures for capturing long-range context. Useful for loss-function ideas (topology-aware losses are relevant to grain/pore continuity).

### 2.2 Materials-SEM-specific segmentation (closest analogs to your task)

This is the more directly relevant cluster — segmentation of particles, grains, and defects in materials-science SEM imagery rather than biological EM:

- **PerovSegNet** (arXiv:2509.26548, 2025) — segments lead iodide, perovskite, and defect domains in solar cell SEM images. Built on an improved YOLOv8x architecture with an Adaptive Shuffle Dilated Convolution Block for multi-scale feature extraction and a Separable Adaptive Downsampling module for boundary recognition, trained on an augmented dataset of 10,994 SEM images, achieving 87.25% mAP with 265.4 GFLOPs while reducing model size and computational load by roughly a quarter relative to baseline YOLOv8x-seg. This is essentially a blueprint for what you're trying to build, down to using YOLOv8-seg as the base and reporting grain-level morphology as the end output.
- **LBMS-SAM** (*ScienceDirect*, 2025) — SAM-guided segmentation for lithium battery material SEM images. Built a 244-image dataset of lithium battery SEM images at 960×600 resolution, with 10,609 annotated masks for training and 2,274 for validation/testing, specifically targeting the high-density, adhesion-heavy particle clusters that cause CNN-based methods to suffer from edge blurring or missed segmentation. Your SEM imagery (powders, particles, porous structures per the NFFA categories) will hit the same overlapping-particle problem.
- **Uncertainty-aware particle segmentation for EM at varied length scales** (*npj Computational Materials*, 2024). Enhances Mask R-CNN to segment particles in SEM images of powder samples, explicitly addressing image blur and particle agglomeration — useful baseline if you want an accuracy-first reference point against which to measure the speed/accuracy tradeoff of a real-time model.
- **Workflow for agglomerated, non-spherical particle segmentation** (*Scientific Reports*, 2021). Flags the core dataset problem directly: there is a severe lack of high-quality annotated image data for training and validating algorithms for nanoparticle segmentation in SEM images, despite how ubiquitous electron microscopy is in materials science — and proposes generating ground truth automatically from paired STEM-in-SEM acquisitions of the same sample area. This is a useful trick if your hardware can acquire STEM-mode images alongside standard SEM.
- **Machine vision-driven particle size/morphology recognition** (*Nanoscale*, RSC, 2020). This paper is the closest published analog to your eventual Phase 2.1 deliverable — it doesn't stop at segmentation, it goes all the way to physical measurement. It performs scale-bar and embedded-text recognition to extract calibration information from the SEM image itself, converting pixel-based size estimates into physical units like µm or nm, and explicitly notes that prior high-throughput methods often failed to measure diameters of overlapping nanoparticles. Read this one closely — it's effectively a worked example of the full pipeline you're describing (segmentation → calibration → PSD).
- **MatSegNet** (arXiv:2312.17251) — carbide precipitate analysis in steels. Demonstrates the standard calibration approach you'll need: carbide sizes are derived from segmentation masks through a calibrated pixel-to-area conversion, where spatial resolution is determined by measuring pixel density along the SEM's own scale bar, yielding the physical area represented by a single pixel.

### 2.3 Foundation models (SAM/SAM2 family) for microscopy

This matters because SAM2 fine-tuning is one of your two realistic architecture paths (Section 3), and there's now a meaningful body of work specifically on adapting it to EM:

- **SAM-EM** (arXiv:2501.03153, 2025) — this is arguably the single most relevant paper to your stated goal, because it explicitly targets real-time EM segmentation. SAM-EM is a domain-adapted foundation model built on SAM2 that unifies segmentation, tracking, and statistical analysis for liquid-phase TEM data, derived through full-model fine-tuning on 46,600 curated synthetic video frames, and integrates particle tracking with statistical tools including mean-squared displacement and particle displacement distribution analysis as part of an end-to-end framework. The pairing of "fine-tuned SAM2 for real-time segmentation" + "statistical particle analysis bolted on afterward" is structurally identical to what you're planning across Phase 2 and Phase 2.1.
- **Lightweight SAM2 fine-tuning for microscopy** (bioRxiv, Nov 2025). Introduces a Colab-based pipeline for fine-tuning SAM2's mask decoder on small, curated microscopy datasets without adding architectural complexity, motivated by the fact that prior adaptations like μSAM and CellSAM require additional transformer or convolutional layers that make them computationally demanding and harder to scale. If your annotated dataset stays small (which is likely — see Section 5), decoder-only fine-tuning is the cheaper, faster-converging path, consistent with your Phase 1 finding that decoder-only fine-tuning of RDUNet/NAFNet outperformed from-scratch training under compute constraints.
- **Foundation models for zero-shot segmentation of scientific images** (arXiv:2506.24039, 2025). Surveys the lightweight SAM variants relevant to a 40ms target: FastSAM (a YOLOv8-based architecture built for efficiency), MobileSAM (optimized for mobile/edge deployment), MedSAM (medical imaging adaptation), and μSAM (microscopy-specific adaptation). FastSAM and MobileSAM are the two variants worth actually benchmarking against YOLO-seg for your latency budget; full SAM2 (even fine-tuned) is unlikely to hit 40ms without aggressive distillation.

### 2.4 Why this matters for your design

A recurring pattern across nearly every materials-SEM segmentation paper above is the same one you already learned the hard way in Phase 1 with noise modeling: **generic, natural-image-trained segmentation models underperform on microscopy until either (a) fine-tuned on domain data, or (b) architecturally adapted for the specific failure mode (overlapping particles, low contrast, fine boundaries)**. None of the papers above achieved strong results from an out-of-the-box COCO/Cityscapes-pretrained model with zero adaptation. This sets expectations for Section 5 (dataset strategy) — you should plan for fine-tuning from the start, not as a contingency.

---

## 3. Methodology and Model Selection

### 3.1 Decision: instance segmentation, not pure semantic segmentation

Your stated downstream goals — particle size distribution, aspect ratio, circularity per particle — require **per-instance masks**, since these are all per-object measurements. Pure semantic segmentation (DeepLabV3+, SegFormer, BiSeNet-class models) only gives you a class map, not separated object identities, so you'd need an additional instance-separation step (e.g., watershed on the semantic mask, which reintroduces the over-segmentation/under-segmentation problems classical methods have struggled with for decades, as seen in the porosity-via-watershed and threshold-based literature in Section 2).

Porosity analysis is the one sub-task where pure semantic segmentation (pore vs. matrix) is actually sufficient and standard — see the ImageJ/thresholding-based porosity workflows in Section 6 — so your architecture should ideally support both a semantic head (for porosity/defect-area) and an instance head (for individual particle morphology), or you run two lightweight models in parallel within the latency budget.

### 3.2 Architecture shortlist

| Candidate | Type | Why it's in the running | Why it might not be the final answer |
|---|---|---|---|
| **YOLOv8-seg / YOLO11-seg** | One-stage instance segmentation | Real-world materials-SEM precedent (PerovSegNet, nanoparticle/STEM pipelines); mature TensorRT export path; smallest realistic gap to 40ms target | Mask quality on heavily overlapping/adjacent particles is the known weak point — exactly the failure mode flagged in the LBMS-SAM and agglomerated-particle papers |
| **Fine-tuned SAM2 (decoder-only)** | Promptable foundation model | SAM-EM is a near-exact precedent for your use case; strong zero-shot generalization before fine-tuning even starts; decoder-only fine-tuning is cheap, consistent with your Phase 1 fine-tuning philosophy | Full SAM2 is a ViT-based encoder — heavier than YOLO-seg; needs prompts (point/box) unless you pair it with an automatic prompt generator, which adds latency and complexity for a fully automated pipeline |
| **MobileSAM / FastSAM** | Distilled/lightweight SAM variants | Designed explicitly for the speed problem that vanilla SAM2 has; FastSAM is itself YOLOv8-based, so it inherits a lot of the same deployment tooling | Less literature precedent specifically on SEM/materials imagery — would need your own validation before trusting it on grain/particle boundaries |
| **Mask R-CNN (ResNet/ConvNeXt backbone)** | Two-stage instance segmentation | Best accuracy ceiling among classic architectures; used in the most accuracy-focused SEM particle papers (uncertainty-aware particle segmentation) | Categorically too slow for 40ms on any of the comparison data found: even on a high-end Titan Xp, Mask R-CNN ran 12.8–15.6 ms per image at fairly low resolution and modest scene complexity — and that gap typically widens at higher resolution and with the GPU contention from a full real-time pipeline. Two-stage region-proposal architectures are structurally harder to push under tight latency budgets than one-stage detectors |
| **DeepLabV3+ / U-Net family (semantic only)** | Semantic segmentation | You already have U-Net-family infrastructure and experience from Phase 1 (RDUNet, Attention U-Net); good fit specifically for the porosity/defect-area task | Doesn't give per-instance masks natively; not a complete answer to the PSD/aspect-ratio/circularity goals on its own |

### 3.3 Recommendation

Given the constraints, a two-track approach is the most defensible:

1. **Primary track — YOLO11-seg (or YOLOv8-seg) fine-tuned on your annotated SEM data**, exported to TensorRT (FP16, and INT8 if accuracy holds up after calibration). This directly targets the instance segmentation + 40ms requirement with the most published precedent in materials SEM specifically (PerovSegNet) and the cleanest path to your latency ceiling (Section 4). This becomes your primary deliverable for "real-time segmentation and detection."

2. **Secondary/parallel track — fine-tuned SAM2 (decoder-only) as an offline-quality reference model**, not necessarily deployed in the real-time path, but used to (a) generate higher-quality pseudo-labels for expanding your training set faster (mirroring the YOLOv8+SAM pseudo-labeling pattern used in the nanoparticle dynamics paper), and (b) serve as an accuracy ceiling to know how much quality you're trading away for speed in the YOLO track.

This mirrors what several of the papers above already did in practice — using YOLO for the fast, deployed detector and SAM as either a refinement step or an annotation accelerator, not as competing alternatives.

A reasonable evaluation matrix going into experiments:
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|               Model                   |                      Expected role                                |                        Target metric                          |
|---------------------------------------|-------------------------------------------------------------------|---------------------------------------------------------------|
|      YOLO11n-seg / YOLOv8n-seg        |               Fastest, deployment baseline                        |                   Latency under load, mAP-mask                |
|      YOLO11s-seg / YOLOv8s-seg        |              Accuracy/speed middle ground                         |                mAP-mask, latency headroom under 40ms          |
| Fine-tuned SAM2 (ViT-B, decoder-only) |         Quality reference / pseudo-label generator                |     mAP-mask, Dice/IoU on hard cases (overlapping particles)  |
|     (Optional) FastSAM/MobileSAM      |       Lightweight SAM alternative for direct comparison           |           Latency vs. YOLO-seg on identical hardware          |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

---

## 4. Hitting the 40ms Ceiling — What the Numbers Actually Say

Your Phase 1 README is explicit that the M3 Max/MPS setup is a real bottleneck and that CUDA + TensorRT is the planned target hardware. The 40ms requirement should be evaluated **only against that target hardware**, because the evidence is unambiguous that it's out of reach on MPS:

- On an RTX 5070 Ti with TensorRT optimization, YOLOv8 segmentation reached up to 374 FPS (roughly 2.7ms/frame) in a real industrial deployment, with TensorRT FP16 providing the dominant share of the speedup over plain ONNX Runtime — for a single-class, fairly simple task (apple counting), so treat this as an optimistic ceiling rather than a guaranteed number for dense multi-particle SEM frames.
- More conservatively, YOLOv8n inference that takes roughly 15–20ms per frame in plain PyTorch typically drops to about 5–8ms with TensorRT FP16 optimization on modern NVIDIA GPUs — this is a more realistic planning number for a small-to-medium YOLO-seg variant on dense, textured SEM imagery.
- On embedded/edge-class hardware rather than a desktop RTX card, YOLOv8s on a Jetson device reached a mean latency around 7.2ms (139 FPS) in FP32 TensorRT, dropping further with INT8 quantization — useful if your eventual SEM control unit needs to run on embedded compute rather than a workstation GPU.
- The honest caveat, directly relevant to your own Phase 1 NAFNet-on-MPS-to-RTX projection: real-world TensorRT latency can diverge meaningfully from vendor documentation depending on GPU variant, driver stack, and export settings — so treat all of the above as directional, not as a guarantee, and budget for your own benchmarking pass before committing to a final model size.

**Practical implication for your 40ms requirement:** on any CUDA + TensorRT target (even a mid-range RTX card, let alone a 4090-class GPU), a YOLO-seg nano or small variant has comfortable headroom under 40ms based on every benchmark surveyed above — the real risk to your latency budget isn't the segmentation model itself, it's everything wrapped around it: image preprocessing, postprocessing (mask upsampling, NMS), and especially the downstream morphometric analysis if it's naively run synchronously in the same loop. Keep the measurement/PSD pipeline (Section 6) decoupled from the real-time segmentation loop — segment in real time, batch the morphometric analysis asynchronously or on a slower cadence, since there is no published precedent for running full PSD/porosity/circularity computation inside a 40ms budget, and forcing it there is an unnecessary self-imposed constraint that doesn't serve the actual goal (a human or downstream system reviewing particle statistics does not need them computed at frame rate).

---

## 5. Dataset Strategy — Where the General Model Ends and Fine-Tuning Begins

This is the part of your plan most likely to be underestimated, because Phase 1 had a genuine advantage that Phase 2 does not: a large, clean, purpose-built dataset (NFFA-Europe) already existed for denoising-style training. Segmentation does not have an equivalent off-the-shelf resource for SEM imagery.

### 5.1 What NFFA-Europe actually gives you (and what it doesn't)

You already used the NFFA-Europe Majority dataset in Phase 1, so it's worth being precise about its actual annotation type, since the framing matters for what's reusable in Phase 2:

- The Majority dataset consists of 21,272 SEM images at 1,024×728 pixels, classified into 10 categories (tips, particles, patterned surfaces, MEMS devices and electrodes, nanowires, porous sponge, biological, powder, films and coated surfaces, and fibres) based on majority agreement among a panel of nanoscientists. No scientific metadata beyond the classification label is attached to the images — meaning **NFFA-Europe has no segmentation masks, no instance annotations, and no bounding boxes**. It is purely an image-classification dataset.
- This means NFFA-Europe is useful for Phase 2 in exactly two ways, neither of which is direct supervised segmentation training: (1) as a large pool of **unlabeled SEM images for self-supervised pretraining** of a backbone before fine-tuning on your small annotated set (this pattern is directly precedented — see the self-supervised SEM particle detection work below), and (2) as a source of category-diverse images for **building your own annotation set** by running auto-labeling tools (SAM-assisted annotation, as several papers below describe) and manually correcting the result.
- One practically useful detail for your eventual calibration step: NFFA-Europe SEM images include a white information bar at the bottom of the frame showing scale and acquisition settings, which is typically cropped out before use in classification/denoising pipelines. For your Phase 2.1 scale-calibration work, this metadata bar (or the equivalent one on your own SEM's image output) is precisely the kind of structured region a scale-bar-reading model (Section 6) would need to parse, rather than discard.

### 5.2 Where general-purpose pretraining can carry you

It's reasonable to start from COCO-pretrained YOLO11-seg / YOLOv8-seg weights (standard Ultralytics releases) and SAM2's publicly released checkpoints, the same way you started Phase 1 fine-tuning from NAFNet's SIDD-pretrained weights and RDUNet's grayscale checkpoint rather than training from scratch. This is justifiable up to the point where the encoder/backbone needs to recognize *low-level visual primitives* — edges, blobs, textures, gradients — which transfer reasonably well from natural images, mirroring your own Phase 1 design decision to freeze RDUNet/NAFNet's encoders and fine-tune only decoders.

**Where general pretraining stops being sufficient:** the moment the task requires distinguishing SEM-specific structures (grain boundaries vs. charging artifacts vs. genuine pore edges, or telling overlapping particles apart from a single fused particle) — none of the natural-image datasets these models were pretrained on contain anything resembling this, and every domain-specific paper surveyed in Section 2.2 needed real fine-tuning data to get usable results, with sample sizes ranging from the low hundreds of images with thousands of annotated masks (LBMS-SAM, PerovSegNet) up to tens of thousands of frames for fully automated pipelines (SAM-EM).

### 5.3 Building your fine-tuning dataset

Given the annotation gap, a realistic dataset-building path, roughly in the order other papers have used it:

1. **Seed set:** A few hundred manually annotated SEM images from material categories that match your actual hardware's likely use cases (particles/powders are the most directly relevant NFFA categories to your domain, given Bharat Atomic's stated semiconductor/materials-science target markets). LBMS-SAM's 244-image, ~13,000-mask dataset is a reasonable scale target for a seed set — small enough to annotate manually with a small team, large enough to fine-tune a decoder.
2. **Annotation tooling:** Don't hand-draw every mask. Using an open-source annotation tool like Label-Studio paired with a SAM-based ML backend for AI-assisted annotation is the standard approach in this literature — annotators correct SAM-generated proposals rather than drawing from scratch, which is dramatically faster.
3. **Pseudo-labeling at scale:** Once the seed model is reasonable, use it (or a YOLOv8+SAM combination, as in the nanoparticle dynamics paper) to auto-generate pseudo-labels across your larger unlabeled pool (NFFA particle/powder categories, plus your own hardware's raw output), and have humans spot-check/correct rather than fully re-annotate. This is the same "5% manually annotated, rest pseudo-labeled" pattern used successfully in the SwinTCN-Seg nanoparticle work.
4. **Domain-specific synthetic augmentation:** You already built a sophisticated SEM noise simulator (`NoiseImage`/`new_augment_sem`) in Phase 1. That same noise model is directly reusable here — apply it to clean, well-segmented reference images to synthetically expand your segmentation training set with realistic SEM degradation, the same logic that made your Phase 1 denoiser generalize. This is a genuine advantage Phase 2 inherits from Phase 1 that most of the papers surveyed didn't have.
5. **Self-supervised pretraining on the unlabeled pool:** if annotation throughput remains a bottleneck, self-supervised learning on a large unlabeled SEM image pool (one cited study curated 25,000 SEM images for this purpose) using architectures like ConvNeXtV2 for particle detection is a documented way to get a stronger starting backbone before your small annotated set ever enters the picture — directly applicable using the full NFFA pool (tens of thousands of images) as the unlabeled corpus.

### 5.4 Annotation scale targets, summarized

| Stage                                      | Approx. scale (from precedent)               | Source                                                                  |
|--------------------------------------------|----------------------------------------------|-------------------------------------------------------------------------|
| Self-supervised backbone pretraining       | ~20,000–25,000 unlabeled images              | Full NFFA-Europe pool; precedent from self-supervised SEM particle work |
| Seed annotated set (manual + SAM-assisted) | 200–250 images, ~10,000–13,000 masks         | LBMS-SAM scale                                                          |
| Pseudo-labeled expansion                   | 5% manual / 95% pseudo-labeled, spot-checked | SwinTCN-Seg / YOLOv8+SAM pattern                                        |
| Synthetic augmentation                     | As large as needed                           | Reuse Phase 1's `NoiseImage` engine on clean references                 |

---

## 6. Phase 2.1 Preview — Where Morphometric Analysis Plugs In

Not the focus of the current phase, but worth scoping now so the segmentation output format doesn't have to be redesigned later. The standard pipeline observed across the porosity/PSD literature is consistent:

1. **Scale calibration:** read the SEM's embedded scale bar or metadata (length in physical units corresponding to a pixel span) — measuring pixel density along the scale bar gives a calibration factor representing the physical length (and by extension, area) of a single pixel. This is the "ask the user for a scale reference" step you described, and it should also attempt automatic extraction from the image's metadata bar when available, falling back to a user-provided value otherwise.
2. **Per-instance measurement from masks:** once masks exist, standard connected-component-style analysis is sufficient — no novel research needed here. Segmented masks are fed into a connected component analysis to compute particle size, shape, and orientation distributions, typically expressed as equivalent spherical diameter. Aspect ratio and circularity are computed directly from each mask's geometric moments/contour, not a separate model.
3. **Porosity specifically** is conventionally a thresholding/segmentation-area ratio problem, not a per-instance one: total porosity is calculated as the area covered by pores above a minimum size threshold (filtering out single-pixel noise), with bin-wise size distributions computed from each individual pore's area, including extrapolation to an equivalent-sphere radius. Your semantic segmentation head's pore class output maps directly onto this without modification.
4. **3D/volume estimates** are the one place where genuine caution is warranted: true volume from a single 2D SEM image is an inherently approximate exercise (you're inferring 3D structure from a 2D projection), and the literature handles it either via stereological approximations (equivalent spherical diameter assumptions, as above) or by requiring actual 3D acquisition (FIB-SEM tomography, serial sectioning) when true volumetry is required. Decide early whether "volume" in your requirements means a stereological estimate from 2D masks (fast, approximate, no new acquisition hardware needed) or genuine 3D reconstruction (accurate, but a materially larger scope than segmentation alone).

---

## 7. Summary of Recommendations

- **Architecture:** YOLO11-seg (or YOLOv8-seg) as the primary real-time deployment model, fine-tuned on domain data and exported to TensorRT FP16/INT8. Fine-tuned SAM2 (decoder-only) as a parallel quality-reference and pseudo-labeling tool, not necessarily a deployed real-time model.
- **Latency:** 40ms is comfortably achievable for the segmentation model itself on CUDA + TensorRT hardware based on every benchmark surveyed; the actual risk is in pipeline overhead and any temptation to run morphometric analysis synchronously in the same loop. Keep PSD/porosity/circularity computation decoupled and asynchronous.
- **Dataset:** NFFA-Europe is classification-only and has no segmentation masks — it's reusable for self-supervised backbone pretraining and as a source pool for SAM-assisted annotation, not for direct supervised segmentation training. Build a seed annotated set (target: low hundreds of images, low thousands of masks), expand via pseudo-labeling, and reuse Phase 1's SEM noise simulator for synthetic augmentation.
- **General-vs-fine-tuned boundary:** general COCO/SA-1B pretrained weights are a legitimate and recommended starting point for low-level visual features (matching your own Phase 1 logic of fine-tuning pretrained NAFNet/RDUNet rather than training from scratch), but every comparable materials-SEM paper required real domain fine-tuning before getting usable segmentation quality — there is no published precedent for a zero-shot, fine-tuning-free approach succeeding on this kind of imagery.
- **Phase 2.1 scoping:** scale calibration from the SEM's own metadata/scale bar, per-instance geometric measurement from masks (standard, not novel), and an early decision on whether "volume" means a 2D stereological estimate or requires true 3D acquisition.

---

## 8. References

1. Aswath, A., et al. (2022). Segmentation in large-scale cellular electron microscopy with deep learning: A literature survey. arXiv:2206.07171.
2. Khadangi, A., Boudier, T., & Rajagopal, V. (2021). EM-stellar: benchmarking deep learning for electron microscopy image segmentation. *Bioinformatics*, 37(1), 97–106.
3. "Deep learning for brain electron microscopy segmentation: Advances, challenges, and future directions in connectomics and ultrastructure analysis." *Computers & Graphics*, 2025.
4. Pan, J. G., Wang, L., & Cai, X. (2025). Automated and Scalable SEM Image Analysis of Perovskite Solar Cell Materials via a Deep Segmentation Framework (PerovSegNet). arXiv:2509.26548.
5. LBMS-SAM: Segment anything model guided SEM image segmentation for lithium battery materials. *ScienceDirect*, 2025.
6. Uncertainty-aware particle segmentation for electron microscopy at varied length scales. *npj Computational Materials*, 2024.
7. Workflow towards automated segmentation of agglomerated, non-spherical particles from electron microscopy images using artificial neural networks. *Scientific Reports*, 2021.
8. Machine vision-driven automatic recognition of particle size and morphology in SEM images. *Nanoscale* (RSC Publishing), 2020. DOI:10.1039/D0NR04140H.
9. MatSegNet: a New Boundary-aware Deep Learning Model for Accurate Carbide Precipitate Analysis in High-Strength Steels. arXiv:2312.17251.
10. SAM-EM: Real-Time Segmentation for Automated Liquid Phase Transmission Electron Microscopy. arXiv:2501.03153, 2025.
11. Lightweight open-source fine-tuning of SAM2 enables domain-specific microscopy segmentation. bioRxiv, 2025.
12. Foundation Models for Zero-Shot Segmentation of Scientific Images without AI-Ready Data. arXiv:2506.24039, 2025.
13. Aversa, R., et al. (2018). The first annotated set of scanning electron microscopy images for nanoscience (NFFA-Europe). *Scientific Data*, 5, 180172.
14. Leveraging unlabeled SEM datasets with self-supervised learning for enhanced particle segmentation. *npj Computational Materials*, 2025.
15. Semi-supervised spatiotemporal segmentation of in situ TEM for nanoparticle dynamics (SwinTCN-Seg). *ScienceDirect*, 2026.
16. Comparing YOLOv8 and Mask R-CNN for instance segmentation in complex orchard environments. arXiv:2312.07935, 2023.
17. Achieving 374 FPS with YOLOv8 Segmentation on NVIDIA RTX 5070 Ti GPU. Medium / cvRealtime, 2026.
18. Evolution of Porosity in Suspension Thermal Sprayed YSZ Thermal Barrier Coatings through Neutron Scattering and Image Analysis Techniques. arXiv:2010.07599.
19. Microstructural investigation of hybrid CAD/CAM restorative dental materials by micro-CT and SEM. arXiv:2308.07341.
