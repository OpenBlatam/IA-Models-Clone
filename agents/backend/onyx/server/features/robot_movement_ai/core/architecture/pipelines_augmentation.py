"""
Data Augmentation Module
=========================

Sistema profesional de data augmentation para deep learning.
Incluye augmentaciones para imágenes, secuencias y datos numéricos.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Union
import numpy as np

try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TORCHVISION_AVAILABLE = False
    torch = None
    transforms = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    Sistema de data augmentation profesional.
    
    Soporta:
    - Augmentación de imágenes
    - Augmentación de secuencias
    - Augmentación de datos numéricos
    - Augmentación de audio
    """
    
    def __init__(self, augmentation_type: str = "image"):
        """
        Inicializar augmentación.
        
        Args:
            augmentation_type: Tipo ("image", "sequence", "numeric", "audio")
        """
        self.augmentation_type = augmentation_type
        self.transforms_list: List[Callable] = []
        logger.info(f"DataAugmentation initialized for {augmentation_type}")
    
    def add_image_transform(
        self,
        transform_type: str,
        **kwargs
    ):
        """
        Agregar transformación de imagen.
        
        Args:
            transform_type: Tipo de transformación
            **kwargs: Parámetros de la transformación
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for image augmentation")
        
        transforms_map = {
            "random_crop": transforms.RandomCrop,
            "random_horizontal_flip": transforms.RandomHorizontalFlip,
            "random_vertical_flip": transforms.RandomVerticalFlip,
            "random_rotation": transforms.RandomRotation,
            "color_jitter": transforms.ColorJitter,
            "random_affine": transforms.RandomAffine,
            "gaussian_blur": transforms.GaussianBlur,
            "random_perspective": transforms.RandomPerspective
        }
        
        if transform_type not in transforms_map:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        transform_class = transforms_map[transform_type]
        transform = transform_class(**kwargs)
        self.transforms_list.append(transform)
        logger.info(f"Added {transform_type} transform")
    
    def add_sequence_augmentation(
        self,
        augmentation_type: str,
        **kwargs
    ):
        """
        Agregar augmentación de secuencias.
        
        Args:
            augmentation_type: Tipo ("noise", "time_shift", "time_stretch", "masking")
            **kwargs: Parámetros
        """
        if augmentation_type == "noise":
            noise_level = kwargs.get("noise_level", 0.01)
            self.transforms_list.append(
                lambda x: x + np.random.normal(0, noise_level, x.shape)
            )
        
        elif augmentation_type == "time_shift":
            shift_range = kwargs.get("shift_range", 0.1)
            def shift_fn(x):
                shift = int(len(x) * shift_range * np.random.uniform(-1, 1))
                return np.roll(x, shift, axis=0)
            self.transforms_list.append(shift_fn)
        
        elif augmentation_type == "masking":
            mask_ratio = kwargs.get("mask_ratio", 0.1)
            def mask_fn(x):
                mask = np.random.random(x.shape) > mask_ratio
                return x * mask
            self.transforms_list.append(mask_fn)
        
        logger.info(f"Added {augmentation_type} augmentation for sequences")
    
    def add_numeric_augmentation(
        self,
        augmentation_type: str,
        **kwargs
    ):
        """
        Agregar augmentación de datos numéricos.
        
        Args:
            augmentation_type: Tipo ("gaussian_noise", "scaling", "rotation")
            **kwargs: Parámetros
        """
        if augmentation_type == "gaussian_noise":
            std = kwargs.get("std", 0.01)
            self.transforms_list.append(
                lambda x: x + np.random.normal(0, std, x.shape)
            )
        
        elif augmentation_type == "scaling":
            scale_range = kwargs.get("scale_range", (0.9, 1.1))
            def scale_fn(x):
                scale = np.random.uniform(scale_range[0], scale_range[1])
                return x * scale
            self.transforms_list.append(scale_fn)
        
        logger.info(f"Added {augmentation_type} augmentation for numeric data")
    
    def apply(self, data: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        """
        Aplicar todas las augmentaciones.
        
        Args:
            data: Datos a augmentar
            
        Returns:
            Datos augmentados
        """
        result = data
        for transform in self.transforms_list:
            result = transform(result)
        return result
    
    def create_compose(self) -> Callable:
        """
        Crear función compose para aplicar múltiples augmentaciones.
        
        Returns:
            Función compose
        """
        if self.augmentation_type == "image" and TORCHVISION_AVAILABLE:
            return transforms.Compose(self.transforms_list)
        else:
            def compose_fn(data):
                result = data
                for transform in self.transforms_list:
                    result = transform(result)
                return result
            return compose_fn


class AdvancedAugmentation:
    """
    Augmentaciones avanzadas usando técnicas modernas.
    
    Incluye:
    - Mixup
    - CutMix
    - AutoAugment
    - RandAugment
    """
    
    @staticmethod
    def mixup(
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
        alpha: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplicar Mixup augmentation.
        
        Args:
            x1, y1: Primer batch
            x2, y2: Segundo batch
            alpha: Parámetro de distribución Beta
            
        Returns:
            Batch mezclado y labels
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        lam = np.random.beta(alpha, alpha)
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y
    
    @staticmethod
    def cutmix(
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplicar CutMix augmentation.
        
        Args:
            x1, y1: Primer batch
            x2, y2: Segundo batch
            alpha: Parámetro de distribución Beta
            
        Returns:
            Batch con CutMix y labels
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        lam = np.random.beta(alpha, alpha)
        
        # Generar máscara rectangular
        b, c, h, w = x1.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        x1[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
        
        # Ajustar lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return x1, mixed_y
    
    @staticmethod
    def create_autoaugment_policy(dataset: str = "imagenet"):
        """
        Crear política AutoAugment.
        
        Args:
            dataset: Dataset ("imagenet", "cifar10", "svhn")
            
        Returns:
            Política de augmentación
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required")
        
        try:
            from torchvision.transforms import AutoAugment, AutoAugmentPolicy
            
            policy_map = {
                "imagenet": AutoAugmentPolicy.IMAGENET,
                "cifar10": AutoAugmentPolicy.CIFAR10,
                "svhn": AutoAugmentPolicy.SVHN
            }
            
            policy = policy_map.get(dataset, AutoAugmentPolicy.IMAGENET)
            return AutoAugment(policy=policy)
        except ImportError:
            logger.warning("AutoAugment not available in this torchvision version")
            return None
    
    @staticmethod
    def create_randaugment(num_ops: int = 2, magnitude: int = 9):
        """
        Crear RandAugment.
        
        Args:
            num_ops: Número de operaciones
            magnitude: Magnitud de augmentación
            
        Returns:
            Transformación RandAugment
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required")
        
        try:
            from torchvision.transforms import RandAugment
            return RandAugment(num_ops=num_ops, magnitude=magnitude)
        except ImportError:
            logger.warning("RandAugment not available in this torchvision version")
            return None

