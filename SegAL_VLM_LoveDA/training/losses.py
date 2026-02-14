import torch
import torch.nn as nn
import torch.nn.functional as F

class SegLoss(nn.Module):
    def __init__(self, num_classes=7, ignore_index=255, label_smoothing=0.0):
        super().__init__()
        try:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=float(label_smoothing))
        except TypeError:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def dice_loss(self, logits, target):
        # target: (B, H, W)
        # logits: (B, C, H, W)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target.clamp(0, self.num_classes-1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Softmax logits
        probs = F.softmax(logits, dim=1)
        
        # Mask out ignore index
        mask = (target != self.ignore_index).unsqueeze(1).float()
        
        intersection = torch.sum(probs * target_one_hot * mask, dim=(2, 3))
        union = torch.sum(probs * mask, dim=(2, 3)) + torch.sum(target_one_hot * mask, dim=(2, 3))
        
        dice = 2.0 * intersection / (union + 1e-6)
        return 1.0 - dice.mean()
        
    def forward(self, logits, target):
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice_loss(logits, target)
        return loss_ce + loss_dice
