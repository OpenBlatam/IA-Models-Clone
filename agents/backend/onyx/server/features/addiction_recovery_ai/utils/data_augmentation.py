"""
Data Augmentation for Recovery Models
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class FeatureAugmentation:
    """Feature augmentation for recovery data"""
    
    @staticmethod
    def add_noise(features: torch.Tensor, noise_level: float = 0.05) -> torch.Tensor:
        """
        Add Gaussian noise to features
        
        Args:
            features: Input features
            noise_level: Noise level (std)
        
        Returns:
            Augmented features
        """
        noise = torch.randn_like(features) * noise_level
        return features + noise
    
    @staticmethod
    def scale_features(features: torch.Tensor, scale_range: tuple = (0.9, 1.1)) -> torch.Tensor:
        """
        Scale features randomly
        
        Args:
            features: Input features
            scale_range: Scale range (min, max)
        
        Returns:
            Augmented features
        """
        scale = torch.empty(features.shape).uniform_(*scale_range)
        return features * scale
    
    @staticmethod
    def mixup(
        features1: torch.Tensor,
        features2: torch.Tensor,
        alpha: float = 0.2
    ) -> torch.Tensor:
        """
        Mixup augmentation
        
        Args:
            features1: First feature set
            features2: Second feature set
            alpha: Mixup alpha parameter
        
        Returns:
            Mixed features
        """
        lam = np.random.beta(alpha, alpha)
        return lam * features1 + (1 - lam) * features2
    
    @staticmethod
    def cutout(features: torch.Tensor, cutout_size: int = 1) -> torch.Tensor:
        """
        Cutout augmentation (zero out random features)
        
        Args:
            features: Input features
            cutout_size: Number of features to zero out
        
        Returns:
            Augmented features
        """
        augmented = features.clone()
        num_features = features.shape[-1]
        
        if cutout_size > 0:
            indices = torch.randperm(num_features)[:cutout_size]
            augmented[..., indices] = 0
        
        return augmented


class SequenceAugmentation:
    """Sequence augmentation for time series data"""
    
    @staticmethod
    def time_shift(sequence: torch.Tensor, shift_range: int = 2) -> torch.Tensor:
        """
        Time shift augmentation
        
        Args:
            sequence: Input sequence
            shift_range: Maximum shift
        
        Returns:
            Augmented sequence
        """
        shift = np.random.randint(-shift_range, shift_range + 1)
        if shift == 0:
            return sequence
        
        shifted = torch.roll(sequence, shift, dims=1)
        return shifted
    
    @staticmethod
    def add_temporal_noise(sequence: torch.Tensor, noise_level: float = 0.05) -> torch.Tensor:
        """
        Add temporal noise
        
        Args:
            sequence: Input sequence
            noise_level: Noise level
        
        Returns:
            Augmented sequence
        """
        noise = torch.randn_like(sequence) * noise_level
        return sequence + noise
    
    @staticmethod
    def mask_timesteps(sequence: torch.Tensor, mask_prob: float = 0.1) -> torch.Tensor:
        """
        Mask random timesteps
        
        Args:
            sequence: Input sequence
            mask_prob: Mask probability
        
        Returns:
            Augmented sequence
        """
        mask = torch.rand(sequence.shape[0], sequence.shape[1], 1) > mask_prob
        return sequence * mask.float()


class AugmentationPipeline:
    """Pipeline for applying multiple augmentations"""
    
    def __init__(self, augmentations: List[Callable], p: float = 0.5):
        """
        Initialize augmentation pipeline
        
        Args:
            augmentations: List of augmentation functions
            p: Probability of applying augmentation
        """
        self.augmentations = augmentations
        self.p = p
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply augmentations"""
        if np.random.random() > self.p:
            return data
        
        augmented = data
        for aug in self.augmentations:
            if np.random.random() < self.p:
                augmented = aug(augmented)
        
        return augmented


def create_feature_augmentation_pipeline(
    noise_level: float = 0.05,
    scale_range: tuple = (0.9, 1.1),
    p: float = 0.5
) -> AugmentationPipeline:
    """Create feature augmentation pipeline"""
    augmentations = [
        lambda x: FeatureAugmentation.add_noise(x, noise_level),
        lambda x: FeatureAugmentation.scale_features(x, scale_range),
        lambda x: FeatureAugmentation.cutout(x, cutout_size=1)
    ]
    return AugmentationPipeline(augmentations, p=p)


def create_sequence_augmentation_pipeline(
    noise_level: float = 0.05,
    shift_range: int = 2,
    mask_prob: float = 0.1,
    p: float = 0.5
) -> AugmentationPipeline:
    """Create sequence augmentation pipeline"""
    augmentations = [
        lambda x: SequenceAugmentation.time_shift(x, shift_range),
        lambda x: SequenceAugmentation.add_temporal_noise(x, noise_level),
        lambda x: SequenceAugmentation.mask_timesteps(x, mask_prob)
    ]
    return AugmentationPipeline(augmentations, p=p)

