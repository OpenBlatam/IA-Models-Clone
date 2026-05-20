"""
Data Transforms - Transformaciones de datos
===========================================

Sistema de transformaciones de datos para preprocessing.
Sigue mejores prácticas de data augmentation y transforms.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available")


class Compose:
    """Componer múltiples transformaciones"""
    
    def __init__(self, transforms: List[Callable]):
        """
        Args:
            transforms: Lista de transformaciones
        """
        self.transforms = transforms
    
    def __call__(self, data: Any) -> Any:
        """Aplicar transformaciones en secuencia"""
        for transform in self.transforms:
            data = transform(data)
        return data


class Normalize:
    """Normalizar tensor"""
    
    def __init__(self, mean: List[float], std: List[float]):
        """
        Args:
            mean: Media por canal
            std: Desviación estándar por canal
        """
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalizar tensor"""
        if tensor.dim() == 3:
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
        
        # Normalize
        normalized = (tensor - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
        
        if normalized.dim() == 4 and normalized.size(0) == 1:
            normalized = normalized.squeeze(0)
        
        return normalized


class ToTensor:
    """Convertir a tensor"""
    
    def __call__(self, data: Any) -> torch.Tensor:
        """Convertir a tensor"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, (list, tuple)):
            return torch.tensor(data)
        elif isinstance(data, torch.Tensor):
            return data
        else:
            try:
                return torch.tensor(data)
            except Exception as e:
                logger.error(f"Error converting to tensor: {e}")
                raise


class RandomAugment:
    """Aumentación aleatoria"""
    
    def __init__(
        self,
        p: float = 0.5,
        augmentations: Optional[List[str]] = None
    ):
        """
        Args:
            p: Probabilidad de aplicar aumentación
            augmentations: Lista de aumentaciones a aplicar
        """
        self.p = p
        self.augmentations = augmentations or ["flip", "rotate", "noise"]
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Aplicar aumentación aleatoria"""
        if torch.rand(1).item() > self.p:
            return tensor
        
        for aug in self.augmentations:
            if aug == "flip" and torch.rand(1).item() < 0.5:
                tensor = torch.flip(tensor, dims=[-1])
            elif aug == "rotate" and torch.rand(1).item() < 0.5:
                angle = torch.randint(-15, 16, (1,)).item()
                # Simple rotation (simplified)
                tensor = tensor  # In production, use proper rotation
            elif aug == "noise" and torch.rand(1).item() < 0.5:
                noise = torch.randn_like(tensor) * 0.1
                tensor = tensor + noise
        
        return tensor


class DataTransformsService:
    """Servicio de transformaciones de datos"""
    
    @staticmethod
    def create_image_transforms(
        size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        augment: bool = False
    ) -> Callable:
        """
        Crear transformaciones para imágenes.
        
        Args:
            size: Tamaño de imagen
            mean: Media para normalización
            std: Desviación estándar para normalización
            augment: Si aplicar aumentación
        
        Returns:
            Función de transformación
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available, using simplified transforms")
        
        transform_list = []
        
        if augment:
            # Training transforms
            if PIL_AVAILABLE:
                transform_list.extend([
                    transforms.RandomResizedCrop(size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ])
            else:
                transform_list.append(RandomAugment(p=0.5))
        else:
            # Validation/test transforms
            if PIL_AVAILABLE:
                transform_list.extend([
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                ])
        
        transform_list.extend([
            ToTensor(),
            Normalize(mean, std),
        ])
        
        return Compose(transform_list)
    
    @staticmethod
    def create_text_transforms(
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ) -> Callable:
        """
        Crear transformaciones para texto.
        
        Args:
            max_length: Longitud máxima
            padding: Si hacer padding
            truncation: Si truncar
        
        Returns:
            Función de transformación
        """
        def transform(text: str) -> Dict[str, Any]:
            # In production, this would use tokenizer
            # For now, simplified version
            return {
                "input_ids": torch.randint(0, 1000, (max_length,)),
                "attention_mask": torch.ones(max_length),
            }
        
        return transform
    
    @staticmethod
    def create_tensor_transforms(
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> Callable:
        """
        Crear transformaciones para tensores.
        
        Args:
            normalize: Si normalizar
            mean: Media (None = calcular automáticamente)
            std: Desviación estándar (None = calcular automáticamente)
        
        Returns:
            Función de transformación
        """
        def transform(tensor: torch.Tensor) -> torch.Tensor:
            tensor = ToTensor()(tensor)
            
            if normalize:
                if mean is None or std is None:
                    # Calculate from tensor
                    mean_val = tensor.mean().item()
                    std_val = tensor.std().item()
                    tensor = (tensor - mean_val) / (std_val + 1e-8)
                else:
                    tensor = Normalize(mean, std)(tensor)
            
            return tensor
        
        return transform




