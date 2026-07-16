"""

WHAT CHANGED FROM THE EARLIER lbms_sam_base.py
------------------------------------------------
The previous version had, per GA branch: Conv2d(1x1) -> BatchNorm -> ReLU
-> WaveletDenoise -> SqueezeExcite, then a 4-way concat -> Conv2d -> BN ->
ReLU -> Conv2d -> BN -> ReLU fusion stack. NONE of that matches the diagram.

The diagram shows, for MDFF specifically:
    GA1, GA2, GA3, GA4 --> DWT --> (per-branch wavelet coefficients,
    e.g. u_l1, u_l2, u_h1, u_h2) --> IDWT --> Post-Processing --> Denoised Feat.

No per-branch conv before DWT. No SqueezeExcite. No 4-way concat-then-conv
fusion -- the diagram's arrows show DWT consuming all 4 GA inputs and
producing fused wavelet sub-bands directly (the IDWT inverts a SINGLE
combined decomposition, not 4 separate ones reconstructed and then
concatenated). "Post-Processing" is one small block after IDWT, not a
per-branch operation.

WHY WE STILL ADD PROJECTION CONVS (per your explicit decision)
------------------------------------------------------------------
You're using SAM2/Hiera as the backbone, not the plain ViT the original
LBMS-SAM paper used. The paper's GA1-GA4 are 4 outputs from the SAME ViT
stack, so they're already the same channel count (e.g. 1024 for ViT-L) --
DWT can consume them directly. Your GA1-GA4 are Hiera STAGE outputs, which
have growing channel counts (144 -> 288 -> 576 -> 1152 for Hiera-L). You
cannot DWT-fuse tensors of different channel counts without projecting them
to a common dimension first. This is a real, necessary deviation forced by
your backbone choice -- not paper-fidelity, and it's documented as such so
nobody mistakes this file for a literal paper reimplementation.

GSEFE matches the diagram closely already (Sobel Conv + Gabor Conv ->
Feature Enhancement -> Edge feat.) -- kept mostly as before, simplified to
match the 2-branch (not 3-way concat with raw gray) structure shown.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_channel_first(x: torch.Tensor) -> torch.Tensor:
    """
    Validate a GA branch tensor is already (B, C, H, W).

    NOTE: this used to be a heuristic reformatter (`_to_channel_first`) that
    guessed BHWC-vs-BCHW by comparing x.shape[-1] against x.shape[1] and
    permuting if the last dim looked smaller. That heuristic is WRONG for
    this codebase: hieradet.py's Hiera.forward() already permutes every
    stage output to (B, C, H, W) internally (`feats = x.permute(0, 3, 1, 2)`,
    hieradet.py line 296) before returning it. Once channel count exceeds
    spatial resolution -- which happens by Hiera's 2nd stage (e.g. 224
    channels at 128x128) -- the old heuristic misread an already-correct
    BCHW tensor as BHWC and silently permuted it into a broken shape. This
    was caught by testing MDFF against real Hiera-shaped tensors, not by
    inspection -- it would have corrupted training silently otherwise.

    Trunk output format is a known, source-confirmed fact, not something to
    infer per-tensor. If you ever swap backbones, update this assumption
    explicitly rather than reintroducing a shape-guessing heuristic.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(x.shape)}")
    return x


def _resize_like(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# GSEFE -- Gabor-Sobel Edge Feature Extraction
# Diagram: image -> [Sobel Conv, Gabor Conv] -> Feature Enhancement -> Edge feat.
# ---------------------------------------------------------------------------

def build_sobel_kernels() -> Tuple[torch.Tensor, torch.Tensor]:
    gx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    gy = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    return gx, gy


def build_gabor_bank(
    thetas: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
    size: int = 5,
    sigma_x: float = 0.5,
    sigma_y: float = 1.5,
    wavelength: float = 4.0,
) -> torch.Tensor:
    half = size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    kernels = []
    for deg in thetas:
        theta = math.pi * deg / 180.0
        x_rot = x * math.cos(theta) + y * math.sin(theta)
        y_rot = -x * math.sin(theta) + y * math.cos(theta)
        gauss = torch.exp(-((x_rot**2) / sigma_x**2 + (y_rot**2) / sigma_y**2))
        kern = gauss * torch.cos(2 * math.pi * x_rot / wavelength)
        kern = kern / (kern.std() + 1e-8)
        kernels.append(kern.view(1, 1, size, size))
    return torch.cat(kernels, dim=0)


class GSEFE(nn.Module):
    """
    Gabor-Sobel Edge Feature Extraction.

    Diagram structure: raw image -> Sobel Conv (fixed kernel) and Gabor Conv
    (fixed kernel bank) run in parallel -> Feature Enhancement (the only
    learnable part) -> Edge feat., at a channel/spatial resolution matching
    mask_feat for the later point-wise product in Feature Fusion.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        self.to_gray = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

        sobel_x, sobel_y = build_sobel_kernels()
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("gabor_bank", build_gabor_bank())  # (4, 1, 5, 5)

        # Feature Enhancement: the diagram's one learnable block, taking the
        # concatenated Sobel + Gabor responses and producing Edge feat.
        # Input channels: 1 (sobel magnitude) + 4 (gabor orientations) = 5
        self.feature_enhancement = nn.Sequential(
            nn.Conv2d(5, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = self.to_gray(image)

        gp = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        sx = F.conv2d(gp, self.sobel_x)
        sy = F.conv2d(gp, self.sobel_y)
        sobel_mag = torch.sqrt(sx.square() + sy.square() + 1e-8)  # (B,1,H,W)

        gp2 = F.pad(gray, (2, 2, 2, 2), mode="reflect")
        gabor_resp = F.conv2d(gp2, self.gabor_bank)  # (B,4,H,W)

        x = torch.cat([sobel_mag, gabor_resp], dim=1)  # (B,5,H,W)
        return self.feature_enhancement(x)


# ---------------------------------------------------------------------------
# Haar DWT / IDWT -- the actual wavelet transform used in the diagram's
# DWT -> (u_l, u_h coefficients) -> IDWT path
# ---------------------------------------------------------------------------

def haar_decompose(x: torch.Tensor):
    """Single-level 2D Haar DWT. Returns (LL, LH, HL, HH)."""
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) / 4.0
    lh = (x00 - x01 + x10 - x11) / 4.0
    hl = (x00 + x01 - x10 - x11) / 4.0
    hh = (x00 - x01 - x10 + x11) / 4.0
    return ll, lh, hl, hh


def haar_reconstruct(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    b, c, h, w = ll.shape
    out = torch.zeros((b, c, 2 * h, 2 * w), device=ll.device, dtype=ll.dtype)
    out[:, :, 0::2, 0::2] = ll + lh + hl + hh
    out[:, :, 0::2, 1::2] = ll - lh + hl - hh
    out[:, :, 1::2, 0::2] = ll + lh - hl - hh
    out[:, :, 1::2, 1::2] = ll - lh - hl + hh
    return out


class SoftThreshold(nn.Module):
    """
    Learnable soft-thresholding on wavelet high-frequency coefficients --
    this is the denoising mechanism (zero out small/noisy high-freq
    coefficients, keep large/structural ones). One threshold parameter per
    channel, shared across all 4 GA branches' high-freq sub-bands since the
    diagram shows a SINGLE shared DWT->...->IDWT path, not 4 independent ones.
    PARAMETERIZATION (post-mortem on the P1 mask-hole regression):
    The original fix here replaced a dead `softplus(zeros_init)` (see git
    history) with `softplus(inverse_softplus(0.05))`, which did make the
    threshold path receive gradient -- but softplus is unbounded above, so a
    single large gradient step (this module has only `channels` scalars, so
    steps are poorly damped relative to a normal conv layer's parameter
    count) can push the threshold arbitrarily high, suppressing real
    high-frequency signal precisely on dense/spiky SEM texture. Post-100-epoch
    checkpoint diffs showed exactly this: all 32 channels per sub-band
    converging into a ~0.04-wide band, i.e. behaving like one shared scalar
    instead of 32 differentiated thresholds.

    Fix: bound the effective threshold to (0, max_threshold) via a scaled
    sigmoid (the "tanh-scaled" reparameterization from the problem log --
    sigmoid is used instead of tanh so the output is naturally non-negative
    without an extra abs()). This caps how much damage one bad gradient step
    can do, and keeps the raw parameter on a comparable scale to the rest of
    the network, so it can share an optimizer/LR schedule sanely (pair with
    a separate low-LR param group anyway -- see TrainingEval -- since this
    is still a very low-dimensional parameter relative to a conv layer).

    A small per-channel init jitter breaks the exact symmetry across
    channels: with all 32 channels starting at bit-identical values and
    seeing highly correlated gradients (they all read off the same summed
    wavelet sub-band), they have little basis to diverge on their own. The
    jitter is small enough not to change the intended starting threshold in
    aggregate.
    """

    def __init__(self, channels: int, init_threshold: float = 0.05, max_threshold: float = 0.5):
        super().__init__()

        if not (0.0 < init_threshold < max_threshold):
            raise ValueError(
                f"init_threshold ({init_threshold}) must be in (0, max_threshold={max_threshold})"
            )
        self.max_threshold = max_threshold

        # inverse sigmoid: solve max_threshold * sigmoid(raw) = init_threshold
        p = init_threshold / max_threshold
        raw_init = math.log(p / (1.0 - p))

        init = torch.full((1, channels, 1, 1), raw_init)
        init = init + torch.randn_like(init) * 0.01  # symmetry-breaking jitter
        self.threshold = nn.Parameter(init)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.max_threshold * torch.sigmoid(self.threshold)
        return torch.sign(x) * F.relu(torch.abs(x) - t)
    

    @torch.no_grad()
    def threshold_diagnostics(self) -> dict:
        """
        Per-channel effective-threshold stats (post-sigmoid, the actual
        values used in forward()). Confirms whether channels have
        differentiated (std rising above the ~0.01 init jitter) or remain
        collapsed into one effective global scalar -- P1's "Not yet done"
        item 1.
        """
        t = (self.max_threshold * torch.sigmoid(self.threshold)).flatten()
        return {
            "mean": t.mean().item(),
            "std": t.std().item(),
            "min": t.min().item(),
            "max": t.max().item(),
        }



class MDFF(nn.Module):
    """
    Multi-scale Denoised Feature Fusion.

    Diagram structure:
        GA1, GA2, GA3, GA4
            --> [projection convs -- OUR ADDITION, see module docstring]
            --> DWT (each branch decomposed; diagram shows 2 high-freq
                lines u_h1/u_h2 and 2 low-freq lines u_l1/u_l2 -- read as:
                the 4 GA branches are fused IN PAIRS at the DWT stage, with
                low-freq coefficients combined and high-freq coefficients
                combined, rather than 4 fully independent decompositions)
            --> soft-threshold denoising on high-freq sub-bands
            --> IDWT (single inverse transform reconstructing one fused map)
            --> Post-Processing (one small conv block)
            --> Denoised Feat.

    We implement this as: project each GA branch to a common channel width,
    sum the 4 projected branches into ONE tensor (this is the most direct
    honest reading of "4 inputs -> 1 DWT -> 1 IDWT" with no per-branch
    fusion step drawn elsewhere in the diagram), DWT-decompose that single
    tensor, soft-threshold its high-freq sub-bands, IDWT back, then run
    Post-Processing.

    NOTE ON FIDELITY: the diagram's u_l1/u_l2/u_h1/u_h2 labeling suggests
    something more granular than a flat sum -- possibly pairwise DWT before
    a second fusion DWT (a 2-level decomposition tree). The paper's full
    methods text (paywalled, not available to me this session) would
    resolve this precisely. What's implemented here is the simplest
    structure consistent with everything VISIBLE in the diagram. Flag this
    as a verify-against-paper-PDF item, not settled fact.
    """

    def __init__(
        self,
        in_dims: Sequence[int],
        out_channels: int = 256,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        if len(in_dims) < 2:
            raise ValueError(f"MDFF expects at least 2 GA feature tensors, got {len(in_dims)}.")
        self.num_levels = len(in_dims)
        self.target_size = target_size

        # OUR ADDITION (not in diagram): per-branch projection to a common
        # channel width, required because Hiera's 4 stage outputs have
        # different channel counts (144/288/576/1152 for Hiera-L), unlike
        # the original paper's single-ViT GA1-GA4 which share one width.
        self.proj = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, bias=False)
            for c in in_dims
        ])

        # Single shared soft-threshold denoiser per high-freq sub-band,
        # matching the diagram's single DWT->...->IDWT path (not 4 parallel
        # WaveletDenoise modules like the earlier, incorrect version had).
        self.denoise_lh = SoftThreshold(out_channels)
        self.denoise_hl = SoftThreshold(out_channels)
        self.denoise_hh = SoftThreshold(out_channels)

        # Post-Processing block (diagram's single labeled box after IDWT)
        self.post_processing = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, ga_features: List[torch.Tensor]) -> torch.Tensor:
        if len(ga_features) != self.num_levels:
            raise ValueError(
                f"MDFF was built for {self.num_levels} feature levels but got "
                f"{len(ga_features)} at forward()."
            )

        ga_cf = [_check_channel_first(x) for x in ga_features]
        target = self.target_size or ga_cf[0].shape[-2:]

        # Project each branch to common width, resize to common spatial
        # size, then sum -> single tensor entering DWT (per diagram's single
        # DWT box consuming all 4 GA arrows).
        fused = None
        for i, ga in enumerate(ga_cf):
            x = self.proj[i](ga)
            x = _resize_like(x, target)
            fused = x if fused is None else fused + x

        # Pad to even spatial dims for clean DWT/IDWT round-trip
        h, w = fused.shape[-2:]
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            fused = F.pad(fused, (0, pad_w, 0, pad_h))

        ll, lh, hl, hh = haar_decompose(fused)
        lh = self.denoise_lh(lh)
        hl = self.denoise_hl(hl)
        hh = self.denoise_hh(hh)
        recon = haar_reconstruct(ll, lh, hl, hh)

        if pad_h or pad_w:
            recon = recon[:, :, :h, :w]

        return self.post_processing(recon)
    
    @torch.no_grad()
    def threshold_diagnostics(self) -> dict:
        """Per-sub-band threshold_diagnostics() -- see SoftThreshold. P1 item 1."""
        return {
            "lh": self.denoise_lh.threshold_diagnostics(),
            "hl": self.denoise_hl.threshold_diagnostics(),
            "hh": self.denoise_hh.threshold_diagnostics(),
        }


# ---------------------------------------------------------------------------
# Feature Fusion -- diagram's final stage
#   (Mask Feat * Output Token) [SAM side]  POINT-WISE PRODUCT  Denoised Feat (from MDFF) fused with Edge feat (from GSEFE)
#
# Reading the diagram precisely: "Feature Fusion" box takes THREE inputs --
# Mask Feat, Output Token (concatenated/combined together first, the small
# vertical bars next to "Output Token" suggest a broadcast/expand op), and
# Denoised Feat (which itself is the MDFF output already informed by
# nothing from GSEFE at that point -- Edge feat. arrives at Feature Fusion
# SEPARATELY per its own arrow, not pre-mixed into Denoised Feat).
# The final circle-dot after Feature Fusion is the POINT-WISE PRODUCT
# between [SAM mask] and [LBMS mask], i.e. between the ORIGINAL SAM Mask
# pathway and the LBMS-SAM refined pathway -- producing LBMS-SAM Mask.
# ---------------------------------------------------------------------------

class FeatureFusion(nn.Module):
    """
    Fuses mask_feat (SAM), denoised_feat (MDFF), edge_feat (GSEFE) into one
    spatial map, then applies SAM's own dynamic-weight dot product using the
    SAME output token (passed through its hypernetwork MLP) that SAM uses
    for its own mask — per the LBMS-SAM diagram.
    """
    def __init__(self, mask_feat_channels: int, token_dim: int = 256):
        super().__init__()
        # mix mask_feat + denoised_feat + edge_feat -> single fused map
        self.mix = nn.Sequential(
            nn.Conv2d(mask_feat_channels * 3, mask_feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mask_feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_feat_channels, mask_feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mask_feat_channels),
            nn.ReLU(inplace=True),
        )
        # SAM's hypernetwork: token -> per-mask dynamic weight vector
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, mask_feat_channels),
        )

    def forward(self, mask_feat, denoised_feat, edge_feat, output_token):
        # all three spatial inputs must already match mask_feat's H, W, C
        # (resizing/projection happens upstream, NOT here)
        fused = torch.cat([mask_feat, denoised_feat, edge_feat], dim=1)
        fused = self.mix(fused)                    # (B, C, H, W)

        weights = self.token_mlp(output_token)      # (B, C)
        mask = torch.einsum('bchw,bc->bhw', fused, weights)  # channel-contracting dot product
        return mask  # (B, H, W) — already at mask_feat's resolution, no upsampling needed


class LBMSSamHead(nn.Module):
    """
    Trainable head combining GSEFE + MDFF + FeatureFusion. This is everything
    in the orange "LBMS-SAM" region of the diagram -- the frozen blue "SAM"
    region (Image Encoder, Mask Decoder) is NOT part of this module; it is
    the SAM2 model you load and freeze separately, then feed into this head.
    """

    def __init__(
        self,
        ga_in_dims:Sequence[int],
        image_channels: int = 3,
        latent_channels: int = 256,
        mdff_target_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.gsefe = GSEFE(in_channels=image_channels, out_channels=latent_channels)
        self.mdff = MDFF(in_dims=ga_in_dims, out_channels=latent_channels, target_size=mdff_target_size)
        self.fusion = FeatureFusion(mask_feat_channels=latent_channels, token_dim=latent_channels)

    def forward(
        self,
        image: torch.Tensor,
        ga_features: List[torch.Tensor],
        mask_feat: torch.Tensor,
        output_token: torch.Tensor,
    ) -> torch.Tensor:
        # (previously: a stray `self.mdff` line here did nothing -- removed)
        denoised_feat = self.mdff(ga_features)
        edge_feat = self.gsefe(image)

        target_hw = mask_feat.shape[-2:]
        if denoised_feat.shape[-2:] != target_hw:
            denoised_feat = F.interpolate(
                denoised_feat, size=target_hw, mode="bilinear", align_corners=False
            )
        if edge_feat.shape[-2:] != target_hw:
            edge_feat = F.interpolate(
                edge_feat, size=target_hw, mode="bilinear", align_corners=False
            )

        # FIX: FeatureFusion.forward's real signature is
        # (mask_feat, denoised_feat, edge_feat, output_token). The old call
        # passed (mask_feat, output_token, denoised_feat, edge_feat) --
        # output_token (shape (B, C)) would have landed in the denoised_feat
        # slot expecting (B, C, H, W), which would crash torch.cat the first
        # time this class was actually used.
        return self.fusion(mask_feat, denoised_feat, edge_feat, output_token)


__all__ = [
    "GSEFE",
    "MDFF",
    "FeatureFusion",
    "LBMSSamHead",
    "SoftThreshold",
    "build_sobel_kernels",
    "build_gabor_bank",
    "haar_decompose",
    "haar_reconstruct",
]