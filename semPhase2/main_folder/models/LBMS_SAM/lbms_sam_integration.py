import torch
import torch.nn as nn
import torch.nn.functional as F

from lbms_sam_base import GSEFE , MDFF, FeatureFusion
from SAM.modeling.sam2_base import SAM2Base
from SAM.sam2_image_predictor import SAM2ImagePredictor
import SAM.modeling.backbones.image_encoder as img_enc
from SAM.modeling.sam2_utils import MLP
from SAM.modeling.backbones.image_encoder import ImageEncoder as image_encoder
from SAM.modeling.sam.mask_decoder import MaskDecoder as mask_decoder


MASK_FEAT_CHANNELS = 32
HIERA_BPLUS_STAGE_CHANNELS = [112, 224, 448, 896]


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
        ):
        
        
        super().__init__()
        self.sam2 = SAM2ImagePredictor(sam2_model)
        self.token_dim = token_dim
        self.mask_feat_channels = mask_feat_channels
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1  # +1 for the single-mask token



        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,)
    

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

    def set_image(self, image):
        """
        image: HWC numpy array (uint8 or float), NOT a tensor.
        Call this once before forward(); forward() no longer calls it itself.
        """
        self.image = image
        self.image_tensor = (
            torch.from_numpy(self.image).permute(2, 0, 1).unsqueeze(0).float()/ 255.0
            )
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

        print("Mask_Channels from SAM2.predict function: ", mask_channels)
        print("Mask Features from SAM2.predict function: ", mask_feats.shape)
        self.mask_channels = mask_channels
        return sam_masks, scores, logits, mask_feats , mask_channels, output_tokens
    
    
    def _get_stability_scores(self, mask_logits: torch.Tensor):
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta  # typically 1.0
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u, 1.0)

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
        
    
        '''MDFF'''
        # FIX #2 continued: pull raw stage outputs from the trunk directly,
        # not the FpnNeck-projected backbone_fpn.

        hierarchical_features = self._trunk_features
        mdff_output = self.mdff(hierarchical_features)
        if mdff_output.shape[-2:] != target_hw:
            mdff_output = F.interpolate(
                mdff_output, size=target_hw, mode="bilinear", align_corners=False
            )

        multimask_output = prompts.get("multimask_output", True)
        tok_range = range(1, self.num_mask_tokens) if multimask_output else range(0, 1)
        delta = 1.0
        lbms_masks_list = []
        lbms_scores_list = []

        '''Feature Fusion'''
        # FIX #5: dropped the stray `mask_feats_channels` int that was being
        # passed as a positional arg (arity mismatch against FeatureFusion's
        # real 4-param signature), and kept argument order matching
        # FeatureFusion.forward(mask_feat, denoised_feat, edge_feat, output_token).



        '''
        WHEN ENOUGH DATA (GROUND TRUTH) IS AVAILABLE, ADD A TRAINABLE IOU HEAD TO PREDICT THE QUALITY OF THE LBMS MASKS, 
        SIMILAR TO SAM'S IOU HEAD. THIS WILL ALLOW US TO SELECT THE BEST MASK AMONG MULTIPLE OUTPUTS.
        self.lbms_iou_head = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()          # output in [0, 1]
            )

        # In forward(), after fusion:
        lbms_score = self.lbms_iou_head(output_tokens_fused).squeeze(-1)  # (B,)
        '''
        
        for tok_idx in tok_range:
            token = output_tokens[:, tok_idx, :]          # (B, 256)
            lbms_mask_i = self.fusion(
                mask_feats, mdff_output, gsefe_output, token
            )                                              # (B, H_latent, W_latent)

            # Stability score per mask
            flat = lbms_mask_i.flatten(-2)                # (B, H*W)
            area_i = (flat > delta).float().sum(-1)
            area_u = (flat > -delta).float().sum(-1)
            score_i = torch.where(area_u > 0, area_i / area_u, torch.ones_like(area_i))

            lbms_masks_list.append(lbms_mask_i)
            lbms_scores_list.append(score_i)
        
        lbms_masks_tensor = torch.stack(lbms_masks_list, dim=1)   # (B, N, H, W)
        lbms_scores_tensor = torch.stack(lbms_scores_list, dim=1) # (B, N)

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



        
        



