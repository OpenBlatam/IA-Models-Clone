"""
Advanced Data Augmentation
==========================

Transformaciones avanzadas de datos para entrenamiento.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    transforms = None

logger = logging.getLogger(__name__)


class AdvancedImageAugmentation:
    """
    Augmentación avanzada de imágenes.
    
    Transformaciones para imágenes de manufactura.
    """
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        brightness_range: tuple = (0.8, 1.2),
        contrast_range: tuple = (0.8, 1.2),
        noise_std: float = 0.05,
        blur_prob: float = 0.3
    ):
        """
        Inicializar augmentación.
        
        Args:
            rotation_range: Rango de rotación (grados)
            brightness_range: Rango de brillo
            contrast_range: Rango de contraste
            noise_std: Desviación estándar de ruido
            blur_prob: Probabilidad de blur
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.blur_prob = blur_prob
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Aplicar augmentación.
        
        Args:
            image: Imagen [C, H, W]
            
        Returns:
            Imagen aumentada
        """
        if not TORCH_AVAILABLE:
            return image
        
        # Rotación aleatoria
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = transforms.functional.rotate(image, angle)
        
        # Ajuste de brillo
        if np.random.random() < 0.5:
            brightness = np.random.uniform(*self.brightness_range)
            image = transforms.functional.adjust_brightness(image, brightness)
        
        # Ajuste de contraste
        if np.random.random() < 0.5:
            contrast = np.random.uniform(*self.contrast_range)
            image = transforms.functional.adjust_contrast(image, contrast)
        
        # Ruido gaussiano
        if np.random.random() < 0.3:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
            image = torch.clamp(image, 0, 1)
        
        # Blur
        if np.random.random() < self.blur_prob:
            kernel_size = np.random.choice([3, 5])
            image = transforms.functional.gaussian_blur(image, kernel_size)
        
        return image


class FeatureAugmentation:
    """
    Augmentación de características numéricas.
    
    Transformaciones para features de manufactura.
    """
    
    def __init__(
        self,
        noise_std: float = 0.02,
        scale_range: tuple = (0.95, 1.05),
        dropout_prob: float = 0.1
    ):
        """
        Inicializar augmentación.
        
        Args:
            noise_std: Desviación estándar de ruido
            scale_range: Rango de escala
            dropout_prob: Probabilidad de dropout de features
        """
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.dropout_prob = dropout_prob
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aplicar augmentación.
        
        Args:
            features: Features [num_features]
            
        Returns:
            Features aumentadas
        """
        if not TORCH_AVAILABLE:
            return features
        
        # Ruido gaussiano
        if np.random.random() < 0.5:
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
        
        # Escala aleatoria
        if np.random.random() < 0.5:
            scale = np.random.uniform(*self.scale_range)
            features = features * scale
        
        # Feature dropout
        if np.random.random() < self.dropout_prob:
            mask = torch.rand_like(features) > 0.1
            features = features * mask
        
        return features


class MixUp:
    """
    MixUp augmentation.
    
    Mezcla dos muestras y sus labels.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Inicializar MixUp.
        
        Args:
            alpha: Parámetro de distribución Beta
        """
        self.alpha = alpha
    
    def __call__(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor
    ) -> tuple:
        """
        Aplicar MixUp.
        
        Args:
            x1: Primera muestra
            y1: Label de primera muestra
            x2: Segunda muestra
            y2: Label de segunda muestra
            
        Returns:
            Tupla (x_mixed, y_mixed)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        
        return x_mixed, y_mixed


class CutMix:
    """
    CutMix augmentation.
    
    Corta y pega regiones entre imágenes.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Inicializar CutMix.
        
        Args:
            alpha: Parámetro de distribución Beta
        """
        self.alpha = alpha
    
    def __call__(
        self,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor
    ) -> tuple:
        """
        Aplicar CutMix.
        
        Args:
            x1: Primera imagen [C, H, W]
            y1: Label de primera imagen
            x2: Segunda imagen [C, H, W]
            y2: Label de segunda imagen
            
        Returns:
            Tupla (x_mixed, y_mixed)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Obtener dimensiones
        _, H, W = x1.shape
        
        # Calcular región a cortar
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Posición aleatoria
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Limites
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Aplicar CutMix
        x_mixed = x1.clone()
        x_mixed[:, bby1:bby2, bbx1:bbx2] = x2[:, bby1:bby2, bbx1:bbx2]
        
        # Ajustar lambda según área real
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_mixed = lam * y1 + (1 - lam) * y2
        
        return x_mixed, y_mixed

Advanced Data Augmentation
==========================

Transformaciones avanzadas de datos para entrenamiento.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    transforms = None

logger = logging.getLogger(__name__)


class AdvancedImageAugmentation:
    """
    Augmentación avanzada de imágenes.
    
    Transformaciones para imágenes de manufactura.
    """
    
    @staticmethod
    def get_training_transforms(
        image_size: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> transforms.Compose:
        """
        Obtener transformaciones de entrenamiento.
        
        Args:
            image_size: Tamaño de imagen
            mean: Media para normalización
            std: Desviación estándar
            
        Returns:
            Transformaciones
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
        ])
    
    @staticmethod
    def get_validation_transforms(
        image_size: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> transforms.Compose:
        """
        Obtener transformaciones de validación.
        
        Args:
            image_size: Tamaño de imagen
            mean: Media
            std: Desviación estándar
            
        Returns:
            Transformaciones
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


class FeatureAugmentation:
    """
    Augmentación de características numéricas.
    
    Transformaciones para features numéricas.
    """
    
    @staticmethod
    def add_noise(features: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Agregar ruido gaussiano.
        
        Args:
            features: Características
            noise_level: Nivel de ruido
            
        Returns:
            Características con ruido
        """
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise
    
    @staticmethod
    def scale_features(features: np.ndarray, scale_range: tuple = (0.95, 1.05)) -> np.ndarray:
        """
        Escalar características.
        
        Args:
            features: Características
            scale_range: Rango de escala
            
        Returns:
            Características escaladas
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return features * scale
    
    @staticmethod
    def mixup(
        features1: np.ndarray,
        features2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray,
        alpha: float = 0.2
    ) -> tuple:
        """
        Mixup augmentation.
        
        Args:
            features1: Características batch 1
            features2: Características batch 2
            labels1: Labels batch 1
            labels2: Labels batch 2
            alpha: Parámetro de mixup
            
        Returns:
            Tupla (features_mixed, labels_mixed)
        """
        lam = np.random.beta(alpha, alpha)
        
        features_mixed = lam * features1 + (1 - lam) * features2
        labels_mixed = lam * labels1 + (1 - lam) * labels2
        
        return features_mixed, labels_mixed


class CutMix:
    """
    CutMix augmentation para imágenes.
    """
    
    @staticmethod
    def cutmix(
        images1: torch.Tensor,
        images2: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor,
        alpha: float = 1.0
    ) -> tuple:
        """
        Aplicar CutMix.
        
        Args:
            images1: Imágenes batch 1
            images2: Imágenes batch 2
            labels1: Labels batch 1
            labels2: Labels batch 2
            alpha: Parámetro de CutMix
            
        Returns:
            Tupla (images_mixed, labels_mixed)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        lam = np.random.beta(alpha, alpha)
        
        batch_size = images1.size(0)
        index = torch.randperm(batch_size).to(images1.device)
        
        # Obtener bounding box
        W, H = images1.size(2), images1.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Aplicar CutMix
        images1[:, :, bbx1:bbx2, bby1:bby2] = images2[index, :, bbx1:bbx2, bby1:bby2]
        
        # Ajustar lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        labels_mixed = lam * labels1 + (1 - lam) * labels2[index]
        
        return images1, labels_mixed

