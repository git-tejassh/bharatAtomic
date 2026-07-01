import torch
import torch.nn as nn
import torch.nn.functional as F

from lbms_sam_base import GSEFE , MDFF, FeatureFusion
from SAM.modeling.sam2_base import SAM2Base
from SAM.sam2_image_predictor import SAM2ImagePredictor
import SAM.modeling.backbones.image_encoder as img_enc


class LBMSSAM2Integration(nn.Module):
    def __init__(self, sam2_model, token_dim: int = 256):
        
        super().__init__()
        self.sam2 = SAM2ImagePredictor(sam2_model)
        self.token_dim = token_dim
        self.enc = img_enc

    def set_image(self, image):
        self.image = image
        self.image_tensor = torch.from_numpy(self.image).permute(2, 0, 1).unsqueeze(0).float()/ 255.0
        self.sam2.set_image(image)
    
    def _get_features(self):
        return self.sam2._features

    def runSAM(self, prompts):
        sam_masks, scores, logits, mask_feats, mask_channels, output_tokens = self.sam2.predict(prompts)
        print(mask_channels)
        self.mask_channels = mask_channels
        return sam_masks, scores, logits, mask_feats , mask_channels, output_tokens
    
    def initFusion(self):
        self.gsefe = GSEFE(in_channels=3, out_channels=self.mask_channels)
        self.mdff = MDFF(out_channels=self.mask_channels)   # adjust to MDFF's actual signature
        self.fusion = FeatureFusion(self.mask_channels, self.token_dim)


    def forward(self, prompts):
        image_tensor = self.image_tensor

    

        '''GSEFE'''
        gsefe_input = image_tensor
        gsefe_output = self.gsefe(gsefe_input)
        print('GSEFE output shape:', gsefe_output.shape)
        
    
        '''MDFF'''
        encoder_output = self.enc(image_tensor)
        hierarchical_features = encoder_output['backbone_fpn']
        mdff_output = self.mdff(hierarchical_features)
        print('MDFF output shape:', mdff_output.shape)

        self.set_image(image_tensor)

        '''MASK Features'''
        sam_masks, scores, logits, mask_feats, mask_channels, output_tokens = self.runSAM(prompts)
        
        mask_feats_channels = mask_feats.shape[1]
        print('MASK Features shape:', mask_feats.shape)

        lbms_mask = self.fusion(mask_feats_channels, mask_feats, mdff_output, gsefe_output, output_tokens)


        return sam_masks, scores, logits, mask_feats, mask_channels, lbms_mask







