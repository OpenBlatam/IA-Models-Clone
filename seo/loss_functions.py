from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta, RAdam, Lion
from torch.optim.lr_scheduler import (
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import math
import numpy as np
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Loss Functions for SEO Service
Comprehensive loss functions and optimization algorithms for SEO tasks
"""

    StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, 
    ReduceLROnPlateau, OneCycleLR, ExponentialLR, MultiStepLR
)

logger = logging.getLogger(__name__)

@dataclass
class LossConfig:
    """Configuration for loss functions"""
    loss_type: str = "cross_entropy"  # cross_entropy, focal, label_smoothing, dice, contrastive, ranking
    alpha: float = 1.0  # Weight for focal loss, label smoothing
    gamma: float = 2.0  # Focusing parameter for focal loss
    smoothing: float = 0.1  # Label smoothing factor
    margin: float = 1.0  # Margin for contrastive/ranking losses
    temperature: float = 0.1  # Temperature for contrastive learning
    class_weights: Optional[torch.Tensor] = None  # Class weights for imbalanced datasets
    reduction: str = "mean"  # mean, sum, none
    ignore_index: int = -100  # Index to ignore in loss computation

@dataclass
class OptimizerConfig:
    """Configuration for optimizers"""
    optimizer_type: str = "adamw"  # adam, adamw, sgd, rmsprop, adagrad, radam, lion
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9  # For Adam variants
    beta2: float = 0.999  # For Adam variants
    eps: float = 1e-8
    amsgrad: bool = False  # For Adam
    nesterov: bool = False  # For SGD
    trust_factor: float = 0.001  # For RAdam
    use_gc: bool = False  # Gradient centralization
    use_lookahead: bool = False  # Lookahead optimization

@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers"""
    scheduler_type: str = "cosine"  # step, cosine, cosine_warm_restarts, reduce_on_plateau, onecycle, exponential
    step_size: int = 30  # For StepLR
    gamma: float = 0.1  # For StepLR, ExponentialLR
    milestones: List[int] = field(default_factory=lambda: [30, 60, 90])  # For MultiStepLR
    T_max: int = 100  # For CosineAnnealingLR
    T_0: int = 10  # For CosineAnnealingWarmRestarts
    T_mult: int = 2  # For CosineAnnealingWarmRestarts
    min_lr: float = 1e-6  # Minimum learning rate
    patience: int = 10  # For ReduceLROnPlateau
    factor: float = 0.5  # For ReduceLROnPlateau
    mode: str = "min"  # For ReduceLROnPlateau
    max_lr: float = 1e-3  # For OneCycleLR
    pct_start: float = 0.3  # For OneCycleLR
    anneal_strategy: str = "cos"  # For OneCycleLR

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in SEO classification tasks"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        
    """__init__ function."""
super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss for better generalization"""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        
    """__init__ function."""
super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss"""
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            smoothed_targets = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        loss = -(smoothed_targets * log_probs).sum(dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    """Dice Loss for segmentation-like tasks in SEO content analysis"""
    
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        
    """__init__ function."""
super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute dice loss"""
        inputs = torch.sigmoid(inputs)
        
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        loss = 1 - dice
        
        return loss

class ContrastiveLoss(nn.Module):
    """Contrastive Loss for learning embeddings in SEO similarity tasks"""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss"""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(embeddings.size(0)).view(-1, 1),
            0
        )
        mask = mask * logits_mask
        
        # Compute positive and negative similarities
        positives = similarity_matrix * mask
        negatives = similarity_matrix * (1 - mask)
        
        # Compute loss
        positives = positives.sum(1, keepdim=True)
        negatives = negatives.max(1, keepdim=True)[0]
        
        loss = torch.clamp(self.margin - positives + negatives, min=0)
        return loss.mean()

class RankingLoss(nn.Module):
    """Ranking Loss for SEO ranking optimization tasks"""
    
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        
    """__init__ function."""
super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ranking loss (pairwise ranking)"""
        # Create pairs of scores and labels
        batch_size = scores.size(0)
        scores_i = scores.unsqueeze(1).expand(batch_size, batch_size)
        scores_j = scores.unsqueeze(0).expand(batch_size, batch_size)
        labels_i = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_j = labels.unsqueeze(0).expand(batch_size, batch_size)
        
        # Create mask for valid pairs (i != j and labels_i > labels_j)
        mask = (labels_i > labels_j).float()
        
        # Compute pairwise loss
        loss = torch.clamp(self.margin - (scores_i - scores_j), min=0)
        loss = loss * mask
        
        if self.reduction == "mean":
            return loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class MultiTaskLoss(nn.Module):
    """Multi-task loss for handling multiple SEO objectives"""
    
    def __init__(self, task_weights: Dict[str, float], task_losses: Dict[str, nn.Module]):
        
    """__init__ function."""
super().__init__()
        self.task_weights = task_weights
        self.task_losses = task_losses
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss"""
        total_loss = 0.0
        
        for task_name, weight in self.task_weights.items():
            if task_name in outputs and task_name in targets:
                task_loss = self.task_losses[task_name](outputs[task_name], targets[task_name])
                total_loss += weight * task_loss
        
        return total_loss

class UncertaintyLoss(nn.Module):
    """Uncertainty-weighted loss for multi-task learning with uncertainty estimation"""
    
    def __init__(self, num_tasks: int):
        
    """__init__ function."""
super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, outputs: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
        """Compute uncertainty-weighted loss"""
        total_loss = 0.0
        
        for i, (output, target) in enumerate(zip(outputs, targets)):
            precision = torch.exp(-self.log_vars[i])
            loss = F.mse_loss(output, target, reduction='none')
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss.mean()

class SEOSpecificLoss(nn.Module):
    """SEO-specific loss combining multiple objectives"""
    
    def __init__(self, 
                 classification_weight: float = 1.0,
                 ranking_weight: float = 0.5,
                 similarity_weight: float = 0.3,
                 content_quality_weight: float = 0.2):
        
    """__init__ function."""
super().__init__()
        self.classification_weight = classification_weight
        self.ranking_weight = ranking_weight
        self.similarity_weight = similarity_weight
        self.content_quality_weight = content_quality_weight
        
        # Initialize sub-losses
        self.classification_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.ranking_loss = RankingLoss(margin=1.0)
        self.similarity_loss = ContrastiveLoss(margin=1.0, temperature=0.1)
        self.content_quality_loss = nn.MSELoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute SEO-specific loss"""
        total_loss = 0.0
        
        # Classification loss (e.g., content type classification)
        if 'classification' in outputs and 'classification_targets' in targets:
            cls_loss = self.classification_loss(outputs['classification'], targets['classification_targets'])
            total_loss += self.classification_weight * cls_loss
        
        # Ranking loss (e.g., search result ranking)
        if 'ranking_scores' in outputs and 'ranking_labels' in targets:
            rank_loss = self.ranking_loss(outputs['ranking_scores'], targets['ranking_labels'])
            total_loss += self.ranking_weight * rank_loss
        
        # Similarity loss (e.g., content similarity)
        if 'embeddings' in outputs and 'similarity_labels' in targets:
            sim_loss = self.similarity_loss(outputs['embeddings'], targets['similarity_labels'])
            total_loss += self.similarity_weight * sim_loss
        
        # Content quality loss (e.g., readability score)
        if 'quality_scores' in outputs and 'quality_targets' in targets:
            qual_loss = self.content_quality_loss(outputs['quality_scores'], targets['quality_targets'])
            total_loss += self.content_quality_weight * qual_loss
        
        return total_loss

class AdvancedOptimizer:
    """Advanced optimizer with gradient centralization and other techniques"""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        if config.optimizer_type == "adam":
            return Adam(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad
            )
        elif config.optimizer_type == "adamw":
            return AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad
            )
        elif config.optimizer_type == "sgd":
            return SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.nesterov
            )
        elif config.optimizer_type == "rmsprop":
            return RMSprop(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                eps=config.eps
            )
        elif config.optimizer_type == "adagrad":
            return Adagrad(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                eps=config.eps
            )
        elif config.optimizer_type == "adadelta":
            return Adadelta(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                eps=config.eps
            )
        elif config.optimizer_type == "radam":
            return RAdam(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                trust_factor=config.trust_factor
            )
        elif config.optimizer_type == "lion":
            return Lion(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

class AdvancedScheduler:
    """Advanced learning rate scheduler with multiple strategies"""
    
    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer, config: SchedulerConfig) -> torch.optim.lr_scheduler._LRScheduler:
        """Create scheduler based on configuration"""
        if config.scheduler_type == "step":
            return StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma
            )
        elif config.scheduler_type == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=config.T_max,
                eta_min=config.min_lr
            )
        elif config.scheduler_type == "cosine_warm_restarts":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.T_0,
                T_mult=config.T_mult,
                eta_min=config.min_lr
            )
        elif config.scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode=config.mode,
                factor=config.factor,
                patience=config.patience,
                min_lr=config.min_lr
            )
        elif config.scheduler_type == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=config.max_lr,
                total_steps=config.T_max,
                pct_start=config.pct_start,
                anneal_strategy=config.anneal_strategy
            )
        elif config.scheduler_type == "exponential":
            return ExponentialLR(
                optimizer,
                gamma=config.gamma
            )
        elif config.scheduler_type == "multistep":
            return MultiStepLR(
                optimizer,
                milestones=config.milestones,
                gamma=config.gamma
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")

class LossFunctionManager:
    """Manager for loss functions and optimization strategies"""
    
    def __init__(self) -> Any:
        self.loss_history = []
        self.optimizer_history = []
        self.scheduler_history = []
    
    def create_loss_function(self, config: LossConfig) -> nn.Module:
        """Create loss function based on configuration"""
        if config.loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(
                weight=config.class_weights,
                reduction=config.reduction,
                ignore_index=config.ignore_index
            )
        elif config.loss_type == "focal":
            return FocalLoss(
                alpha=config.alpha,
                gamma=config.gamma,
                reduction=config.reduction
            )
        elif config.loss_type == "label_smoothing":
            return LabelSmoothingLoss(
                smoothing=config.smoothing,
                reduction=config.reduction
            )
        elif config.loss_type == "dice":
            return DiceLoss(reduction=config.reduction)
        elif config.loss_type == "contrastive":
            return ContrastiveLoss(
                margin=config.margin,
                temperature=config.temperature
            )
        elif config.loss_type == "ranking":
            return RankingLoss(
                margin=config.margin,
                reduction=config.reduction
            )
        elif config.loss_type == "seo_specific":
            return SEOSpecificLoss()
        else:
            raise ValueError(f"Unsupported loss type: {config.loss_type}")
    
    def create_optimizer_and_scheduler(self, model: nn.Module, 
                                     optimizer_config: OptimizerConfig,
                                     scheduler_config: SchedulerConfig) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Create optimizer and scheduler pair"""
        optimizer = AdvancedOptimizer.create_optimizer(model, optimizer_config)
        scheduler = AdvancedScheduler.create_scheduler(optimizer, scheduler_config)
        
        # Record configuration
        self.optimizer_history.append({
            'type': optimizer_config.optimizer_type,
            'lr': optimizer_config.learning_rate,
            'config': optimizer_config
        })
        
        self.scheduler_history.append({
            'type': scheduler_config.scheduler_type,
            'config': scheduler_config
        })
        
        return optimizer, scheduler
    
    def get_loss_summary(self) -> Dict[str, Any]:
        """Get summary of loss functions used"""
        return {
            'total_losses': len(self.loss_history),
            'loss_types': [loss['type'] for loss in self.loss_history],
            'optimizer_types': [opt['type'] for opt in self.optimizer_history],
            'scheduler_types': [sch['type'] for sch in self.scheduler_history]
        }

# Utility functions
def create_seo_loss_function(loss_type: str = "seo_specific", **kwargs) -> nn.Module:
    """Create SEO-specific loss function"""
    config = LossConfig(loss_type=loss_type, **kwargs)
    manager = LossFunctionManager()
    return manager.create_loss_function(config)

def create_seo_optimizer(model: nn.Module, optimizer_type: str = "adamw", **kwargs) -> torch.optim.Optimizer:
    """Create SEO-optimized optimizer"""
    config = OptimizerConfig(optimizer_type=optimizer_type, **kwargs)
    return AdvancedOptimizer.create_optimizer(model, config)

def create_seo_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str = "cosine", **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """Create SEO-optimized scheduler"""
    config = SchedulerConfig(scheduler_type=scheduler_type, **kwargs)
    return AdvancedScheduler.create_scheduler(optimizer, config)

# Example usage
if __name__ == "__main__":
    # Create sample model and data
    model = nn.Linear(100, 10)
    inputs = torch.randn(32, 100)
    targets = torch.randint(0, 10, (32,))
    
    # Test different loss functions
    print("=== Testing Loss Functions ===")
    
    # Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    loss = focal_loss(model(inputs), targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Label Smoothing Loss
    label_smooth_loss = LabelSmoothingLoss(smoothing=0.1)
    loss = label_smooth_loss(model(inputs), targets)
    print(f"Label Smoothing Loss: {loss.item():.4f}")
    
    # Test optimizers
    print("\n=== Testing Optimizers ===")
    
    optimizers = ["adam", "adamw", "sgd", "rmsprop", "radam"]
    for opt_type in optimizers:
        optimizer = create_seo_optimizer(model, optimizer_type=opt_type)
        print(f"{opt_type.upper()} optimizer created successfully")
    
    # Test schedulers
    print("\n=== Testing Schedulers ===")
    
    optimizer = create_seo_optimizer(model, optimizer_type="adamw")
    schedulers = ["step", "cosine", "cosine_warm_restarts", "reduce_on_plateau"]
    
    for sch_type in schedulers:
        scheduler = create_seo_scheduler(optimizer, scheduler_type=sch_type)
        print(f"{sch_type.upper()} scheduler created successfully") 