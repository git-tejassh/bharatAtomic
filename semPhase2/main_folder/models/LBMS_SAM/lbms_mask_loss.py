import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss

class LBMS_MaskLoss(nn.Module):
    """
    Composite mask loss: MONAI's Dice+Focal, plus a separate BCE term.
 
    Expects:
        pred_logits: (B, 1, H, W) raw logits (pre-sigmoid)
        gt_mask:     (B, 1, H, W) binary {0, 1} float mask
 
    No boundary term by default -- add one only if Boundary IoU on val
    shows it's actually needed; distance-transform-based boundary losses
    are expensive to recompute every step and shouldn't be a default cost.
    """

    def __init__(self, lambda_dice: float = 1.0, lambda_focal: float = 1.0,
                 lambda_bce: float = 1.0, focal_gamma: float = 2.0):
        super().__init__()
        self.dice_focal = DiceFocalLoss(
            sigmoid=True,             # our pred is raw logits, not already-activated probs
            gamma=focal_gamma,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_bce = lambda_bce
 
    def forward(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        if pred_logits.shape != gt_mask.shape:
            raise ValueError(f"shape mismatch: pred {pred_logits.shape} vs gt {gt_mask.shape}")
        dice_focal_loss = self.dice_focal(pred_logits, gt_mask)
        bce_loss = self.bce(pred_logits, gt_mask)
        return dice_focal_loss + self.lambda_bce * bce_loss