"""
Data Augmentation Service - Aumento de datos
=============================================

Sistema para aumentar datasets usando técnicas avanzadas.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class AugmentationConfig:
    """Configuración de aumentación"""
    # Image augmentations
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotation: float = 0.0  # degrees
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    # Text augmentations
    synonym_replacement: bool = False
    random_insertion: bool = False
    random_deletion: bool = False
    back_translation: bool = False
    # Noise
    add_noise: bool = False
    noise_std: float = 0.1
    # Mixup/Cutout
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutout: bool = False
    cutout_holes: int = 1
    cutout_length: int = 16


class DataAugmentationService:
    """Servicio de aumentación de datos"""
    
    def __init__(self):
        """Inicializar servicio"""
        logger.info("DataAugmentationService initialized")
    
    def augment_image(
        self,
        image: np.ndarray,
        config: AugmentationConfig
    ) -> np.ndarray:
        """Aumentar imagen"""
        augmented = image.copy()
        
        # Horizontal flip
        if config.horizontal_flip and np.random.random() > 0.5:
            augmented = np.fliplr(augmented)
        
        # Vertical flip
        if config.vertical_flip and np.random.random() > 0.5:
            augmented = np.flipud(augmented)
        
        # Rotation
        if config.rotation > 0:
            angle = np.random.uniform(-config.rotation, config.rotation)
            # Simple rotation (would use scipy or PIL in production)
            # For now, just return original
            pass
        
        # Brightness
        if config.brightness > 0:
            factor = 1 + np.random.uniform(-config.brightness, config.brightness)
            augmented = np.clip(augmented * factor, 0, 255 if augmented.dtype == np.uint8 else 1.0)
        
        # Add noise
        if config.add_noise:
            noise = np.random.normal(0, config.noise_std, augmented.shape)
            augmented = augmented + noise
            augmented = np.clip(augmented, 0, 255 if augmented.dtype == np.uint8 else 1.0)
        
        return augmented
    
    def mixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.2
    ) -> tuple:
        """Aplicar Mixup augmentation"""
        if len(X) < 2:
            return X, y
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation
        index = np.random.permutation(len(X))
        
        # Mixup
        mixed_X = lam * X + (1 - lam) * X[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_X, mixed_y
    
    def cutout(
        self,
        image: np.ndarray,
        num_holes: int = 1,
        length: int = 16
    ) -> np.ndarray:
        """Aplicar Cutout augmentation"""
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        for _ in range(num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0
        
        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        return image * mask
    
    def augment_text(
        self,
        text: str,
        config: AugmentationConfig
    ) -> str:
        """Aumentar texto"""
        augmented = text
        
        # Random deletion
        if config.random_deletion and np.random.random() > 0.5:
            words = augmented.split()
            if len(words) > 1:
                # Delete random word
                idx = np.random.randint(len(words))
                words.pop(idx)
                augmented = " ".join(words)
        
        # Random insertion (simplified)
        if config.random_insertion and np.random.random() > 0.5:
            words = augmented.split()
            if len(words) > 0:
                # Insert random word (would use synonym in production)
                idx = np.random.randint(len(words) + 1)
                words.insert(idx, "word")  # Placeholder
                augmented = " ".join(words)
        
        return augmented
    
    def create_augmentation_pipeline(
        self,
        config: AugmentationConfig
    ) -> Optional[Any]:
        """Crear pipeline de aumentación para imágenes"""
        if not TORCH_AVAILABLE:
            return None
        
        transforms_list = []
        
        if config.horizontal_flip:
            transforms_list.append(transforms.RandomHorizontalFlip())
        
        if config.rotation > 0:
            transforms_list.append(transforms.RandomRotation(config.rotation))
        
        if config.brightness > 0 or config.contrast > 0 or config.saturation > 0:
            transforms_list.append(transforms.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
            ))
        
        if transforms_list:
            return transforms.Compose(transforms_list)
        
        return None
    
    def augment_dataset(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        config: AugmentationConfig,
        num_augmented: int = 1
    ) -> tuple:
        """Aumentar dataset completo"""
        augmented_X = [X]
        augmented_y = [y] if y is not None else None
        
        for _ in range(num_augmented):
            if config.use_mixup and y is not None:
                X_aug, y_aug = self.mixup(X, y, config.mixup_alpha)
                augmented_X.append(X_aug)
                if augmented_y is not None:
                    augmented_y.append(y_aug)
            else:
                # Apply individual augmentations
                X_aug = np.array([self.augment_image(img, config) for img in X])
                augmented_X.append(X_aug)
                if augmented_y is not None:
                    augmented_y.append(y)
        
        final_X = np.concatenate(augmented_X, axis=0)
        final_y = np.concatenate(augmented_y, axis=0) if augmented_y is not None else None
        
        return final_X, final_y




