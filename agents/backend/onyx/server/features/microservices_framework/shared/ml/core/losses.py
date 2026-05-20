"""
Custom Loss Functions
Additional loss functions for various tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Target labels [batch_size]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Target labels [batch_size]
            
        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            true_dist[targets == self.ignore_index] = 0
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute dice loss.
        
        Args:
            inputs: Predictions [batch_size, num_classes, ...]
            targets: Ground truth [batch_size, ...]
            
        Returns:
            Dice loss
        """
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Compute dice coefficient
        intersection = (inputs_flat * targets_flat).sum(dim=2)
        union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for representation learning.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim] (optional)
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Positive pair distance
        pos_dist = F.pairwise_distance(anchor, positive)
        
        if negative is not None:
            negative = F.normalize(negative, p=2, dim=1)
            neg_dist = F.pairwise_distance(anchor, negative)
            
            # Contrastive loss
            loss = torch.mean(
                torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
            )
        else:
            # InfoNCE-style contrastive loss
            # Compute similarity matrix
            batch_size = anchor.size(0)
            labels = torch.arange(batch_size).to(anchor.device)
            
            # Concatenate anchor and positive
            features = torch.cat([anchor, positive], dim=0)
            
            # Compute similarity
            similarity_matrix = torch.matmul(features, features.T) / self.temperature
            
            # Mask for positive pairs
            mask = torch.eye(batch_size * 2, dtype=torch.bool).to(anchor.device)
            similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))
            
            # Compute loss
            loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss



