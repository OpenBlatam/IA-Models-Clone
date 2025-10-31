from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import logging
import numpy as np
from typing import Any, List, Dict, Optional
import asyncio
"""
Loss Functions for HeyGen AI.

Advanced loss functions for deep learning models including classification,
regression, segmentation, and custom loss functions following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


class ClassificationLosses:
    """Classification loss functions."""

    @staticmethod
    def cross_entropy_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """Cross entropy loss with label smoothing.

        Args:
            predictions: Model predictions (logits).
            targets: Target labels.
            weight: Class weights.
            ignore_index: Index to ignore.
            reduction: Reduction method.
            label_smoothing: Label smoothing factor.

        Returns:
            torch.Tensor: Computed loss.
        """
        return F.cross_entropy(
            predictions, targets, weight, ignore_index, reduction, label_smoothing
        )

    @staticmethod
    def focal_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Focal loss for handling class imbalance.

        Args:
            predictions: Model predictions (logits).
            targets: Target labels.
            alpha: Alpha parameter for class balancing.
            gamma: Gamma parameter for focusing.
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed focal loss.
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        if reduction == "mean":
            return focal_loss.mean()
        elif reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    @staticmethod
    def dice_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-6,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Dice loss for segmentation tasks.

        Args:
            predictions: Model predictions (probabilities).
            targets: Target labels.
            smooth: Smoothing factor.
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed dice loss.
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten predictions and targets
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (predictions_flat * targets_flat).sum()
        dice_coefficient = (2.0 * intersection + smooth) / (
            predictions_flat.sum() + targets_flat.sum() + smooth
        )
        
        dice_loss = 1 - dice_coefficient
        
        if reduction == "mean":
            return dice_loss.mean()
        elif reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss

    @staticmethod
    def iou_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-6,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """IoU (Intersection over Union) loss.

        Args:
            predictions: Model predictions (probabilities).
            targets: Target labels.
            smooth: Smoothing factor.
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed IoU loss.
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten predictions and targets
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (predictions_flat * targets_flat).sum()
        union = predictions_flat.sum() + targets_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        iou_loss = 1 - iou
        
        if reduction == "mean":
            return iou_loss.mean()
        elif reduction == "sum":
            return iou_loss.sum()
        else:
            return iou_loss

    @staticmethod
    def hinge_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        margin: float = 1.0,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Hinge loss for binary classification.

        Args:
            predictions: Model predictions (logits).
            targets: Target labels (0 or 1).
            margin: Margin parameter.
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed hinge loss.
        """
        # Convert targets to -1 and 1
        targets_binary = 2 * targets - 1
        
        hinge_loss = torch.clamp(margin - predictions * targets_binary, min=0)
        
        if reduction == "mean":
            return hinge_loss.mean()
        elif reduction == "sum":
            return hinge_loss.sum()
        else:
            return hinge_loss


class RegressionLosses:
    """Regression loss functions."""

    @staticmethod
    def mse_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Mean squared error loss.

        Args:
            predictions: Model predictions.
            targets: Target values.
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed MSE loss.
        """
        return F.mse_loss(predictions, targets, reduction=reduction)

    @staticmethod
    def mae_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Mean absolute error loss.

        Args:
            predictions: Model predictions.
            targets: Target values.
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed MAE loss.
        """
        return F.l1_loss(predictions, targets, reduction=reduction)

    @staticmethod
    def huber_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        delta: float = 1.0,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Huber loss for robust regression.

        Args:
            predictions: Model predictions.
            targets: Target values.
            delta: Delta parameter.
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed Huber loss.
        """
        return F.smooth_l1_loss(predictions, targets, reduction=reduction, beta=delta)

    @staticmethod
    def log_cosh_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Log-cosh loss for regression.

        Args:
            predictions: Model predictions.
            targets: Target values.
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed log-cosh loss.
        """
        diff = predictions - targets
        log_cosh = torch.log(torch.cosh(diff))
        
        if reduction == "mean":
            return log_cosh.mean()
        elif reduction == "sum":
            return log_cosh.sum()
        else:
            return log_cosh

    @staticmethod
    def quantile_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantile: float = 0.5,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Quantile loss for regression.

        Args:
            predictions: Model predictions.
            targets: Target values.
            quantile: Quantile parameter (0-1).
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed quantile loss.
        """
        diff = predictions - targets
        quantile_loss = torch.where(
            diff >= 0,
            quantile * diff,
            (quantile - 1) * diff
        )
        
        if reduction == "mean":
            return quantile_loss.mean()
        elif reduction == "sum":
            return quantile_loss.sum()
        else:
            return quantile_loss


class SegmentationLosses:
    """Segmentation loss functions."""

    @staticmethod
    def bce_dice_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """Combined BCE and Dice loss.

        Args:
            predictions: Model predictions (logits).
            targets: Target labels.
            bce_weight: Weight for BCE loss.
            dice_weight: Weight for Dice loss.
            smooth: Smoothing factor.

        Returns:
            torch.Tensor: Combined loss.
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets)
        
        # Dice loss
        predictions_sigmoid = torch.sigmoid(predictions)
        predictions_flat = predictions_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (predictions_flat * targets_flat).sum()
        dice_coefficient = (2.0 * intersection + smooth) / (
            predictions_flat.sum() + targets_flat.sum() + smooth
        )
        dice_loss = 1 - dice_coefficient
        
        # Combined loss
        combined_loss = bce_weight * bce_loss + dice_weight * dice_loss
        
        return combined_loss

    @staticmethod
    def focal_dice_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 2.0,
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """Combined Focal and Dice loss.

        Args:
            predictions: Model predictions (logits).
            targets: Target labels.
            alpha: Alpha parameter for focal loss.
            gamma: Gamma parameter for focal loss.
            smooth: Smoothing factor.

        Returns:
            torch.Tensor: Combined loss.
        """
        # Focal loss
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # Dice loss
        predictions_sigmoid = torch.sigmoid(predictions)
        predictions_flat = predictions_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (predictions_flat * targets_flat).sum()
        dice_coefficient = (2.0 * intersection + smooth) / (
            predictions_flat.sum() + targets_flat.sum() + smooth
        )
        dice_loss = 1 - dice_coefficient
        
        # Combined loss
        combined_loss = focal_loss + dice_loss
        
        return combined_loss

    @staticmethod
    def tversky_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """Tversky loss for segmentation.

        Args:
            predictions: Model predictions (logits).
            targets: Target labels.
            alpha: Alpha parameter (weight for false positives).
            beta: Beta parameter (weight for false negatives).
            smooth: Smoothing factor.

        Returns:
            torch.Tensor: Computed Tversky loss.
        """
        predictions_sigmoid = torch.sigmoid(predictions)
        predictions_flat = predictions_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (predictions_flat * targets_flat).sum()
        fps = (predictions_flat * (1 - targets_flat)).sum()
        fns = ((1 - predictions_flat) * targets_flat).sum()
        
        tversky_coefficient = (intersection + smooth) / (
            intersection + alpha * fps + beta * fns + smooth
        )
        tversky_loss = 1 - tversky_coefficient
        
        return tversky_loss


class CustomLosses:
    """Custom loss functions."""

    @staticmethod
    def contrastive_loss(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 1.0,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """Contrastive loss for learning embeddings.

        Args:
            embeddings: Learned embeddings.
            labels: Class labels.
            margin: Margin for negative pairs.
            temperature: Temperature parameter.

        Returns:
            torch.Tensor: Computed contrastive loss.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings)
        
        # Create mask for positive pairs
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Positive pairs loss
        positive_distances = distances[labels_matrix]
        positive_loss = positive_distances.mean()
        
        # Negative pairs loss
        negative_distances = distances[~labels_matrix]
        negative_loss = torch.clamp(margin - negative_distances, min=0).mean()
        
        return positive_loss + negative_loss

    @staticmethod
    def triplet_loss(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float = 1.0
    ) -> torch.Tensor:
        """Triplet loss for learning embeddings.

        Args:
            anchor: Anchor embeddings.
            positive: Positive embeddings.
            negative: Negative embeddings.
            margin: Margin parameter.

        Returns:
            torch.Tensor: Computed triplet loss.
        """
        # Compute distances
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        triplet_loss = torch.clamp(pos_distance - neg_distance + margin, min=0)
        
        return triplet_loss.mean()

    @staticmethod
    def cosine_embedding_loss(
        input1: torch.Tensor,
        input2: torch.Tensor,
        target: torch.Tensor,
        margin: float = 0.0,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """Cosine embedding loss.

        Args:
            input1: First input tensor.
            input2: Second input tensor.
            target: Target labels (1 for similar, -1 for dissimilar).
            margin: Margin parameter.
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed cosine embedding loss.
        """
        return F.cosine_embedding_loss(input1, input2, target, margin, reduction)

    @staticmethod
    def kl_divergence_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """KL divergence loss.

        Args:
            predictions: Predicted log probabilities.
            targets: Target probabilities.
            reduction: Reduction method.

        Returns:
            torch.Tensor: Computed KL divergence loss.
        """
        return F.kl_div(predictions, targets, reduction=reduction)

    @staticmethod
    def multi_task_loss(
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float]
    ) -> torch.Tensor:
        """Multi-task loss combining multiple loss functions.

        Args:
            predictions: Dictionary of predictions for each task.
            targets: Dictionary of targets for each task.
            loss_weights: Dictionary of weights for each task.

        Returns:
            torch.Tensor: Combined multi-task loss.
        """
        total_loss = 0.0
        
        for task_name in predictions.keys():
            if task_name in targets and task_name in loss_weights:
                pred = predictions[task_name]
                target = targets[task_name]
                weight = loss_weights[task_name]
                
                # Choose appropriate loss function based on task
                if task_name.startswith("classification"):
                    task_loss = F.cross_entropy(pred, target)
                elif task_name.startswith("regression"):
                    task_loss = F.mse_loss(pred, target)
                elif task_name.startswith("segmentation"):
                    task_loss = F.binary_cross_entropy_with_logits(pred, target)
                else:
                    task_loss = F.mse_loss(pred, target)
                
                total_loss += weight * task_loss
        
        return total_loss


class LossFunctionFactory:
    """Factory for creating loss functions."""

    @staticmethod
    def create_loss_function(
        loss_type: str,
        **kwargs
    ) -> Callable:
        """Create loss function.

        Args:
            loss_type: Type of loss function.
            **kwargs: Additional arguments.

        Returns:
            Callable: Loss function.

        Raises:
            ValueError: If loss type is not supported.
        """
        if loss_type == "cross_entropy":
            return lambda pred, target: ClassificationLosses.cross_entropy_loss(
                pred, target, **kwargs
            )
        elif loss_type == "focal":
            return lambda pred, target: ClassificationLosses.focal_loss(
                pred, target, **kwargs
            )
        elif loss_type == "dice":
            return lambda pred, target: ClassificationLosses.dice_loss(
                pred, target, **kwargs
            )
        elif loss_type == "iou":
            return lambda pred, target: ClassificationLosses.iou_loss(
                pred, target, **kwargs
            )
        elif loss_type == "hinge":
            return lambda pred, target: ClassificationLosses.hinge_loss(
                pred, target, **kwargs
            )
        elif loss_type == "mse":
            return lambda pred, target: RegressionLosses.mse_loss(
                pred, target, **kwargs
            )
        elif loss_type == "mae":
            return lambda pred, target: RegressionLosses.mae_loss(
                pred, target, **kwargs
            )
        elif loss_type == "huber":
            return lambda pred, target: RegressionLosses.huber_loss(
                pred, target, **kwargs
            )
        elif loss_type == "log_cosh":
            return lambda pred, target: RegressionLosses.log_cosh_loss(
                pred, target, **kwargs
            )
        elif loss_type == "quantile":
            return lambda pred, target: RegressionLosses.quantile_loss(
                pred, target, **kwargs
            )
        elif loss_type == "bce_dice":
            return lambda pred, target: SegmentationLosses.bce_dice_loss(
                pred, target, **kwargs
            )
        elif loss_type == "focal_dice":
            return lambda pred, target: SegmentationLosses.focal_dice_loss(
                pred, target, **kwargs
            )
        elif loss_type == "tversky":
            return lambda pred, target: SegmentationLosses.tversky_loss(
                pred, target, **kwargs
            )
        elif loss_type == "contrastive":
            return lambda emb, labels: CustomLosses.contrastive_loss(
                emb, labels, **kwargs
            )
        elif loss_type == "triplet":
            return lambda anchor, pos, neg: CustomLosses.triplet_loss(
                anchor, pos, neg, **kwargs
            )
        elif loss_type == "cosine_embedding":
            return lambda input1, input2, target: CustomLosses.cosine_embedding_loss(
                input1, input2, target, **kwargs
            )
        elif loss_type == "kl_divergence":
            return lambda pred, target: CustomLosses.kl_divergence_loss(
                pred, target, **kwargs
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    @staticmethod
    def get_loss_config(loss_type: str) -> Dict[str, Any]:
        """Get default configuration for loss function.

        Args:
            loss_type: Type of loss function.

        Returns:
            Dict[str, Any]: Default configuration.
        """
        configs = {
            "cross_entropy": {
                "weight": None,
                "ignore_index": -100,
                "reduction": "mean",
                "label_smoothing": 0.0
            },
            "focal": {
                "alpha": 1.0,
                "gamma": 2.0,
                "reduction": "mean"
            },
            "dice": {
                "smooth": 1e-6,
                "reduction": "mean"
            },
            "iou": {
                "smooth": 1e-6,
                "reduction": "mean"
            },
            "hinge": {
                "margin": 1.0,
                "reduction": "mean"
            },
            "mse": {
                "reduction": "mean"
            },
            "mae": {
                "reduction": "mean"
            },
            "huber": {
                "delta": 1.0,
                "reduction": "mean"
            },
            "log_cosh": {
                "reduction": "mean"
            },
            "quantile": {
                "quantile": 0.5,
                "reduction": "mean"
            },
            "bce_dice": {
                "bce_weight": 0.5,
                "dice_weight": 0.5,
                "smooth": 1e-6
            },
            "focal_dice": {
                "alpha": 1.0,
                "gamma": 2.0,
                "smooth": 1e-6
            },
            "tversky": {
                "alpha": 0.3,
                "beta": 0.7,
                "smooth": 1e-6
            },
            "contrastive": {
                "margin": 1.0,
                "temperature": 0.1
            },
            "triplet": {
                "margin": 1.0
            },
            "cosine_embedding": {
                "margin": 0.0,
                "reduction": "mean"
            },
            "kl_divergence": {
                "reduction": "mean"
            }
        }
        
        return configs.get(loss_type, {})


def create_loss_function(loss_type: str, **kwargs) -> Callable:
    """Factory function to create loss function.

    Args:
        loss_type: Type of loss function.
        **kwargs: Additional arguments.

    Returns:
        Callable: Created loss function.
    """
    return LossFunctionFactory.create_loss_function(loss_type, **kwargs) 