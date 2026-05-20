"""
Data Augmentation
Modular data augmentation utilities for image datasets
"""

import torch
import torchvision.transforms as transforms
from typing import Optional, List, Callable
from PIL import Image
import numpy as np


class AugmentationBuilder:
    """
    Builder for creating augmentation pipelines
    """
    
    @staticmethod
    def get_train_transforms(
        image_size: int = 224,
        mean: List[float] = None,
        std: List[float] = None,
        use_color_jitter: bool = True,
        use_random_erasing: bool = False,
        use_cutout: bool = False,
    ) -> transforms.Compose:
        """
        Get training augmentation transforms
        
        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
            use_color_jitter: Use color jitter
            use_random_erasing: Use random erasing
            use_cutout: Use cutout augmentation
            
        Returns:
            Composed transforms
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet mean
        if std is None:
            std = [0.229, 0.224, 0.225]  # ImageNet std
        
        transform_list = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        if use_color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                )
            )
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        if use_random_erasing:
            transform_list.append(
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
            )
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_val_transforms(
        image_size: int = 224,
        mean: List[float] = None,
        std: List[float] = None,
    ) -> transforms.Compose:
        """
        Get validation/test augmentation transforms
        
        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Composed transforms
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    @staticmethod
    def get_test_time_augmentation(
        image_size: int = 224,
        num_augmentations: int = 5,
        mean: List[float] = None,
        std: List[float] = None,
    ) -> List[transforms.Compose]:
        """
        Get test-time augmentation transforms
        
        Args:
            image_size: Target image size
            num_augmentations: Number of augmentations
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            List of transform pipelines
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        base_transforms = [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        
        aug_transforms = []
        
        # Original
        aug_transforms.append(transforms.Compose(base_transforms))
        
        # Horizontal flip
        flip_transforms = base_transforms.copy()
        flip_transforms.insert(-2, transforms.RandomHorizontalFlip(p=1.0))
        aug_transforms.append(transforms.Compose(flip_transforms))
        
        # Additional augmentations if needed
        for i in range(num_augmentations - 2):
            aug_list = [
                transforms.Resize(int(image_size * 1.14)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
            aug_transforms.append(transforms.Compose(aug_list))
        
        return aug_transforms[:num_augmentations]


class MixUp:
    """
    MixUp augmentation
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> tuple:
        """
        Apply MixUp to batch
        
        Args:
            batch_x: Input batch
            batch_y: Label batch
            
        Returns:
            Mixed batch and labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    """
    CutMix augmentation
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> tuple:
        """
        Apply CutMix to batch
        
        Args:
            batch_x: Input batch
            batch_y: Label batch
            
        Returns:
            Mixed batch and labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(batch_x.size(), lam)
        batch_x[:, :, bbx1:bbx2, bby1:bby2] = batch_x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size()[-1] * batch_x.size()[-2]))
        
        y_a, y_b = batch_y, batch_y[index]
        
        return batch_x, y_a, y_b, lam
    
    def _rand_bbox(self, size: tuple, lam: float) -> tuple:
        """Generate random bounding box"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2



