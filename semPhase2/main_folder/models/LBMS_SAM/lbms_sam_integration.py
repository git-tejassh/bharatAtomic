from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from lbms_sam_base import GSEFE , MDFF, FeatureFusion
from SAM.modeling.sam2_base import SAM2Base
from SAM.sam2_image_predictor import SAM2ImagePredictor
import SAM.modeling.backbones.image_encoder as img_enc
from SAM.modeling.sam2_utils import MLP
from SAM.modeling.backbones.image_encoder import ImageEncoder as image_encoder
from SAM.modeling.sam.mask_decoder import MaskDecoder as mask_decoder
from dataclasses import dataclass


MASK_FEAT_CHANNELS = 32
HIERA_BPLUS_STAGE_CHANNELS = [112, 224, 448, 896]


@dataclass
class LBMSTrainOutput:
    """
    forward_train()'s return type. `masks`/`iou_pred` are named to match
    what TrainingEval.combined_loss (lbms_sam_main.py) reads via attribute
    access -- unlike forward()'s inference-only return tuple, everything
    here is still attached to the autograd graph (no .detach()/.numpy()).
    """
    masks: torch.Tensor        # (B, N, H, W) LBMS mask logits, gt-resolution
    iou_pred: torch.Tensor     # (B, N) LBMS stability scores
    sam_masks: torch.Tensor    # (B, N, H, W) frozen SAM masks (logits or bool), for reference
    sam_scores: torch.Tensor   # (B, N) frozen SAM iou predictions
    mask_feats: torch.Tensor   # (B, C, h, w) SAM decoder's upscaled mask features


class LBMSSAM2Integration(nn.Module):
    def __init__(self,
        sam2_model,
        token_dim: int = 256,
        ga_in_dims: list = None,
        mask_feat_channels: int = MASK_FEAT_CHANNELS,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        iou_prediction_use_sigmoid=False,
        transformer_dim: int = 256,
        num_multimask_outputs: int = 3,
        target_size: int = 1024,
        use_mdff: bool = True,
        use_gsefe: bool = True,
        ):
        
        
        super().__init__()
        self.sam2 = SAM2ImagePredictor(sam2_model)
        self.token_dim = token_dim
        self.mask_feat_channels = mask_feat_channels
        self.num_multimask_outputs = num_multimask_outputs
        self.target_size = target_size  
        self.num_mask_tokens = num_multimask_outputs + 1  # +1 for the single-mask token

        # P1/P5 ablation switches: isolate which branch (if either) causes
        # the mask-hole regression by zeroing its contribution to fusion.
        # Both branches still run (so shapes/checkpoints stay identical
        # regardless of flags) -- only their OUTPUT is zeroed, which also
        # zeroes their gradient, so this doubles as an ablation-training
        # knob, not just an eval-time one.
        self.use_mdff = use_mdff
        self.use_gsefe = use_gsefe



        # self.iou_prediction_head = MLP(
        #     transformer_dim,
        #     iou_head_hidden_dim,
        #     self.num_mask_tokens,
        #     iou_head_depth,
        #     sigmoid_output=iou_prediction_use_sigmoid,)
    

        # self.gsefe = GSEFE(in_channels=3, out_channels=self.mask_feat_channels)
        # self.mdff = MDFF(out_channels=self.mask_feat_channels)   # adjust to MDFF's actual signature
        # self.fusion = FeatureFusion(self.mask_feat_channels, self.token_dim)

        # FIX #1 (root cause of the AttributeError): construct the trainable
        # adapter modules exactly ONCE, here in __init__, like any other
        # nn.Module submodule. Not in a separate initFusion() that nothing
        # calls, and not per-forward-call based on a value (mask_channels)
        # that's only discovered mid-forward. mask_feat_channels is a fixed
        # architectural constant (32), not something to rediscover per image.

        ga_in_dims = ga_in_dims or HIERA_BPLUS_STAGE_CHANNELS
        self.gsefe = GSEFE(in_channels=3, out_channels=mask_feat_channels)
        self.mdff = MDFF(in_dims=ga_in_dims, out_channels=mask_feat_channels)
        self.fusion = FeatureFusion(mask_feat_channels, token_dim)

        # FIX #2: self.enc was the raw imported *module*
        # (`SAM.modeling.backbones.image_encoder`), which is not callable.
        # We need the actual encoder instance living on the SAM2 model, and
        # specifically its .trunk submodule -- NOT the neck's backbone_fpn
        # output. backbone_fpn is uniformly `d_model` (256) channels at every
        # level (see FpnNeck: every level goes through a Conv2d projecting to
        # d_model) -- it is NOT the varying per-stage channel counts
        # (112/224/448/896) that MDFF's in_dims assumes. The trunk's raw
        # stage outputs are what we actually want.

        self.image_encoder = self.sam2.model.image_encoder
        
        self.lbms_iou_head = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()          # output in [0, 1]
            )

    def _preprocess_gsefe_input(self, image_np):
        """
        Replicate LBMSCocoDataset.__getitem__'s IMAGE preprocessing exactly so
        GSEFE sees the same spatial frame at inference as in training: resize
        the longest side to target_size, pad to a target_size square, scale to
        [0,1], move to device. NOT SAM-normalized -- GSEFE consumes raw [0,1],
        matching forward_train's `images`. Fixes both the frame mismatch and
        the device bug in one place.
        """
        orig_h, orig_w = image_np.shape[:2]
        scale = self.target_size / max(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(image_np, (new_w, new_h))  # cv2 takes (W, H)
        padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        return (
            torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ).to(self.sam2.device)
    
    def set_image(self, image):
        """
        image: HWC numpy array (uint8 or float), NOT a tensor.
        Call this once before forward(); forward() no longer calls it itself.
        """
        self.image = image
        self.image_tensor = self._preprocess_gsefe_input(image)
        self.sam2.set_image(image)

        # cache trunk outputs here --
        with torch.no_grad():
            mdff_input = self.sam2._transforms(image)
            mdff_input = mdff_input[None, ...].to(self.sam2.device)
            self._trunk_features = self.image_encoder.trunk(mdff_input)
    
    def _get_features(self):
        return self.sam2._features

    def runSAM(self, prompts:dict):
         # FIX #3: predict() takes individual kwargs (point_coords=,
        # point_labels=, multimask_output=, ...), not one dict positionally.
        # `self.sam2.predict(prompts)` was binding the whole dict to the
        # `point_coords` parameter, leaving point_labels=None, which trips
        # the `assert point_labels is not None` inside _prep_prompts.

        sam_masks, scores, logits, mask_feats, mask_channels, output_tokens = (
            self.sam2.predict(**prompts,)
        )

        # print("Mask_Channels from SAM2.predict function: ", mask_channels)
        # print("Mask Features from SAM2.predict function: ", mask_feats.shape)
        self.mask_channels = mask_channels
        return sam_masks, scores, logits, mask_feats , mask_channels, output_tokens
    
    
    # def _get_stability_scores(self, mask_logits: torch.Tensor):
    #     mask_logits = mask_logits.flatten(-2)
    #     stability_delta = self.dynamic_multimask_stability_delta  # typically 1.0
    #     area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
    #     area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
    #     return torch.where(area_u > 0, area_i / area_u, 1.0)
    

    def _fuse_lbms_masks(self, mask_feats, mdff_output, gsefe_output, output_tokens, multimask_output):
        """
        Shared by forward() (inference) and forward_train(): runs the
        trainable GSEFE/MDFF/FeatureFusion path over each requested SAM
        output token. Returns (masks, stability_scores), both still
        attached to the autograd graph -- callers decide whether/when to
        detach.
        """
        tok_range = range(1, self.num_mask_tokens) if multimask_output else range(0, 1)
        delta = 1.0
        lbms_masks_list = []
        lbms_scores_list = []

        for tok_idx in tok_range:
            token = output_tokens[:, tok_idx, :]          # (B, 256)
            lbms_mask_i = self.fusion(
                mask_feats, mdff_output, gsefe_output, token
            )                                              # (B, H_latent, W_latent)

            # Stability score per mask
            flat = lbms_mask_i.flatten(-2)                # (B, H*W)
            area_i = (flat > delta).float().sum(-1)
            area_u = (flat > -delta).float().sum(-1)
            score_i = self.lbms_iou_head(token).squeeze(-1)

            lbms_masks_list.append(lbms_mask_i)
            lbms_scores_list.append(score_i)

        lbms_masks_tensor = torch.stack(lbms_masks_list, dim=1)   # (B, N, H, W)
        lbms_scores_tensor = torch.stack(lbms_scores_list, dim=1) # (B, N)
        return lbms_masks_tensor, lbms_scores_tensor
    
    @torch.no_grad()
    def compute_batch_iou_targets(pred_masks_logits, gt_masks, threshold=0.0):
        """
        pred_masks_logits: (B, N, H, W) LBMS mask logits (gt-resolution, from LBMSTrainOutput.masks)
        gt_masks:          (B, N, H, W) or (B, 1, H, W) broadcastable binary ground truth
        Returns: (B, N) real IoU per candidate mask, detached -- this is a target, not a loss term.
        """
        pred_bin = (pred_masks_logits > threshold).float()
        intersection = (pred_bin * gt_masks).sum(dim=(-2, -1))
        union = ((pred_bin + gt_masks) > 0).float().sum(dim=(-2, -1))
        return torch.where(union > 0, intersection / union, torch.zeros_like(intersection))

    def forward_train(self, images, point_coords, point_labels, multimask_output=True):
        """
        Training-mode forward path. Unlike forward(), this:
          - takes batched tensors directly (no set_image() call, no
            SAM2ImagePredictor numpy/single-image wrapper),
          - never detaches or converts to numpy, so gradients reach
            self.gsefe / self.mdff / self.fusion via loss.backward().

        images:       (B, 3, H, W) float tensor in [0, 1], NOT SAM-normalized
                       (this is the raw dataloader image -- GSEFE consumes it
                       directly, same as forward()'s self.image_tensor).
        point_coords: (B, N, 2) float tensor, pixel coords in `images`' frame.
        point_labels: (B, N) int tensor (1 = fg, 0 = bg, -1 = pad).

        Returns an LBMSTrainOutput. The frozen SAM encoder/prompt-encoder/
        mask-decoder path runs under torch.no_grad() (mirroring
        SAM2ImagePredictor.set_image()/predict(), which are @torch.no_grad()
        themselves) -- only GSEFE/MDFF/FeatureFusion, which own trainable
        parameters, run with grad tracking enabled.
        """
        device = self.sam2.device
        images = images.to(device)
        orig_hw = tuple(images.shape[-2:])

        with torch.no_grad():
            # Batched equivalent of SAM2ImagePredictor.set_image(): resize +
            # normalize, run the image encoder, and reshape backbone feats --
            # SAM2ImagePredictor itself only exposes a batch-size-1 API via
            # set_image(), but the underlying model methods are batch-native.
            input_image = self.sam2._transforms.transforms(images)

            backbone_out = self.sam2.model.forward_image(input_image)
            _, vision_feats, _, _ = self.sam2.model._prepare_backbone_features(backbone_out)
            if self.sam2.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.sam2.model.no_mem_embed

            feats = [
                feat.permute(1, 2, 0).view(images.shape[0], -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self.sam2._bb_feat_sizes[::-1])
            ][::-1]
            image_embed, high_res_feats = feats[-1], feats[:-1]

            # Raw per-stage trunk features for MDFF (see FIX #2 in __init__ --
            # NOT the neck-projected backbone_fpn).
            hierarchical_features = self.image_encoder.trunk(input_image)

            point_coords_t = point_coords.to(device).float()
            point_labels_t = point_labels.to(device).int()
            unnorm_coords = self.sam2._transforms.transform_coords(
                point_coords_t, normalize=True, orig_hw=orig_hw
            )

            sparse_embeddings, dense_embeddings = self.sam2.model.sam_prompt_encoder(
                points=(unnorm_coords, point_labels_t),
                boxes=None,
                masks=None,
            )

            decoder_out = self.sam2.model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=False,  # images are already batched, one point-set each
                high_res_features=(
                    high_res_feats if self.sam2.model.use_high_res_features_in_sam else None
                ),
            )

        sam_masks = decoder_out.masks
        sam_scores = decoder_out.iou_pred
        mask_feats = decoder_out.mask_feat
        output_tokens = decoder_out.output_tokens
        target_hw = mask_feats.shape[-2:]

        '''GSEFE'''
        gsefe_output = self.gsefe(images)
        if gsefe_output.shape[-2:] != target_hw:
            gsefe_output = F.interpolate(
                gsefe_output, size=target_hw, mode="bilinear", align_corners=False
            )

        if not self.use_gsefe:
            gsefe_output = torch.zeros_like(gsefe_output)


        '''MDFF'''
        mdff_output = self.mdff(hierarchical_features)
        if mdff_output.shape[-2:] != target_hw:
            mdff_output = F.interpolate(
                mdff_output, size=target_hw, mode="bilinear", align_corners=False
            )
        if not self.use_mdff:
            mdff_output = torch.zeros_like(mdff_output)


        '''Feature Fusion'''
        lbms_masks_tensor, lbms_scores_tensor = self._fuse_lbms_masks(
            mask_feats, mdff_output, gsefe_output, output_tokens, multimask_output
        )

        lbms_mask_upscaled = self.sam2._transforms.postprocess_masks(
            lbms_masks_tensor, orig_hw
        )  # (B, N, H, W) logits, still on the autograd graph

        return LBMSTrainOutput(
            masks=lbms_mask_upscaled,
            iou_pred=lbms_scores_tensor,
            sam_masks=sam_masks,
            sam_scores=sam_scores,
            mask_feats=mask_feats,
        )

    def forward(self, prompts):
        image_tensor = self.image_tensor

        # FIX #4: removed the stray `self.set_image(image_tensor)` call that
        # used to sit here. It passed an already-built tensor into a method
        # that immediately does torch.from_numpy() on it (crash), and it was
        # redundant anyway -- set_image() must already have been called
        # externally before forward(), since image_tensor is read above.
 
        # Run SAM first. We need mask_feats' spatial resolution (H, W) to
        # correctly resize GSEFE/MDFF outputs before fusion -- doing SAM
        # first also removes the ordering problem that made initFusion()
        # unreachable in the old code.

        sam_masks, scores, logits, mask_feats, mask_channels, output_tokens = (
            self.runSAM(prompts)
        )
        target_hw = mask_feats.shape[-2:]

        '''GSEFE'''
        gsefe_output = self.gsefe(image_tensor)
        if gsefe_output.shape[-2:] != target_hw:
            gsefe_output = F.interpolate(
                gsefe_output, size=target_hw, mode="bilinear", align_corners=False
            )
        
        if not self.use_gsefe:
            gsefe_output = torch.zeros_like(gsefe_output)
    
        '''MDFF'''
        # FIX #2 continued: pull raw stage outputs from the trunk directly,
        # not the FpnNeck-projected backbone_fpn.

        hierarchical_features = self._trunk_features
        mdff_output = self.mdff(hierarchical_features)
        if mdff_output.shape[-2:] != target_hw:
            mdff_output = F.interpolate(
                mdff_output, size=target_hw, mode="bilinear", align_corners=False
            )
        if not self.use_mdff:
            mdff_output = torch.zeros_like(mdff_output)

        multimask_output = prompts.get("multimask_output", True)
        '''
        WHEN ENOUGH DATA (GROUND TRUTH) IS AVAILABLE, ADD A TRAINABLE IOU HEAD TO PREDICT THE QUALITY OF THE LBMS MASKS, 
        SIMILAR TO SAM'S IOU HEAD. THIS WILL ALLOW US TO SELECT THE BEST MASK AMONG MULTIPLE OUTPUTS.
        '''
        

        
    

        lbms_masks_tensor, lbms_scores_tensor = self._fuse_lbms_masks(
            mask_feats, mdff_output, gsefe_output, output_tokens, multimask_output
        )


        lbms_mask_upscaled = self.sam2._transforms.postprocess_masks(
            lbms_masks_tensor,
            self.sam2._orig_hw[-1],
        )  

        


        lbms_mask_np = (
            (lbms_mask_upscaled > self.sam2.mask_threshold)
            .squeeze(0)
            .float()
            .detach()
            .cpu()
            .numpy()
            )  # (N, H_orig, W_orig) — same format as SAM's masks_np
        
        lbms_score_np = (
            lbms_scores_tensor.squeeze(0).detach().cpu().numpy()
            )  # (N,)
 
        return sam_masks, scores, logits, mask_feats, mask_channels, lbms_mask_np, lbms_score_np



        



