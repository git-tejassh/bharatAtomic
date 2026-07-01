import torch
import torch.nn as nn
import torch.nn.functional as F

from lbms_sam_base import GSEFE , MDFF, FeatureFusion
from SAM.modeling.sam2_base import SAM2Base
from SAM.sam2_image_predictor import SAM2ImagePredictor


class LBMSSAM2Integration(nn.Module):
    def __init__(self, sam2_model, feature_dim = 256):
        self.encoder = self.SAM.modeling.backbones.image_encoder
        super().__init__()
        self.sam2 = SAM2ImagePredictor(sam2_model)


    def runSAM(self, prompts):
        sam_masks, scores, logits, mask_feats, output_tokens = self.sam2.predict(prompts)
        return sam_masks, scores, logits, mask_feats , output_tokens
    

    def forward(self, image_tensor, prompts):
        '''GSEFE'''
        gsefe_input = image_tensor
        gsefe_output = GSEFE(image = gsefe_input)
        
    
        '''MDFF'''
        encoder_output = self.encoder(image_tensor)
        hierarchical_features = encoder_output['backbone_fpn']
        mdff_output = MDFF(hierarchical_features)

        '''MASK Features'''
        sam_masks, scores, logits, mask_feats, output_tokens = self.runSAM(prompts)
        mask_feats_channels = mask_feats.shape[1]
        lbms_mask = FeatureFusion(mask_feats_channels, mask_feats, mdff_output, gsefe_output, output_tokens)

        return sam_masks, scores, logits, mask_feats, lbms_mask







