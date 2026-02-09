"""
Data Augmentation - Modular Augmentation
========================================

Augmentación modular de datos para entrenamiento.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import torch
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Augmenter(ABC):
    """Clase base para augmentadores."""
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Aplicar augmentación."""
        pass


class GaussianNoiseAugmenter(Augmenter):
    """Augmentación con ruido gaussiano."""
    
    def __init__(self, std: float = 0.01):
        """
        Inicializar augmentador.
        
        Args:
            std: Desviación estándar del ruido
        """
        self.std = std
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Aplicar ruido gaussiano."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        noise = torch.randn_like(data) * self.std
        return data + noise


class RandomScaleAugmenter(Augmenter):
    """Augmentación con escalado aleatorio."""
    
    def __init__(self, scale_range: tuple = (0.9, 1.1)):
        """
        Inicializar augmentador.
        
        Args:
            scale_range: Rango de escalado (min, max)
        """
        self.scale_range = scale_range
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Aplicar escalado aleatorio."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        return data * scale


class RandomShiftAugmenter(Augmenter):
    """Augmentación con desplazamiento aleatorio."""
    
    def __init__(self, shift_range: tuple = (-0.1, 0.1)):
        """
        Inicializar augmentador.
        
        Args:
            shift_range: Rango de desplazamiento (min, max)
        """
        self.shift_range = shift_range
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Aplicar desplazamiento aleatorio."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        shift = torch.empty(data.shape[-1]).uniform_(*self.shift_range)
        return data + shift.unsqueeze(0)


class RandomRotationAugmenter(Augmenter):
    """Augmentación con rotación aleatoria."""
    
    def __init__(self, angle_range: tuple = (-15, 15)):
        """
        Inicializar augmentador.
        
        Args:
            angle_range: Rango de ángulos en grados (min, max)
        """
        self.angle_range = angle_range
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Aplicar rotación aleatoria."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        # Rotación simple en 2D (XY plane)
        angle = torch.empty(1).uniform_(*self.angle_range).item()
        angle_rad = np.deg2rad(angle)
        
        if data.shape[-1] >= 2:
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rotation_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=data.dtype)
            
            # Aplicar rotación
            if data.dim() == 2:
                return torch.matmul(data, rotation_matrix[:data.shape[-1], :data.shape[-1]].T)
            else:
                return torch.matmul(data, rotation_matrix[:data.shape[-1], :data.shape[-1]].T)
        
        return data


class TimeWarpAugmenter(Augmenter):
    """Augmentación con time warping."""
    
    def __init__(self, sigma: float = 0.2, knot: int = 4):
        """
        Inicializar augmentador.
        
        Args:
            sigma: Desviación estándar para warping
            knot: Número de knots
        """
        self.sigma = sigma
        self.knot = knot
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Aplicar time warping."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        orig_steps = data.shape[0]
        
        # Generar knots aleatorios
        random_warps = torch.randn(self.knot) * self.sigma
        warp_steps = torch.linspace(0, orig_steps - 1, self.knot + 2)[1:-1]
        warp_steps = warp_steps + random_warps
        
        # Interpolación
        from scipy.interpolate import interp1d
        
        if isinstance(data, torch.Tensor):
            data_np = data.numpy()
        else:
            data_np = data
        
        warped_data = []
        for dim in range(data_np.shape[-1]):
            original_steps = np.arange(orig_steps)
            warped_steps = np.concatenate([[0], warp_steps.numpy(), [orig_steps - 1]])
            warped_values = np.concatenate([[0], random_warps.numpy(), [0]])
            
            interp_func = interp1d(warped_steps, warped_values, kind='cubic')
            warp = interp_func(original_steps)
            
            warped_dim = data_np[:, dim] + warp
            warped_data.append(warped_dim)
        
        warped_data = np.stack(warped_data, axis=-1)
        return torch.from_numpy(warped_data) if isinstance(data, torch.Tensor) else warped_data


class ComposeAugmenter(Augmenter):
    """Composición de múltiples augmentadores."""
    
    def __init__(self, augmenters: List[Augmenter], p: float = 1.0):
        """
        Inicializar composición.
        
        Args:
            augmenters: Lista de augmentadores
            p: Probabilidad de aplicar augmentación
        """
        self.augmenters = augmenters
        self.p = p
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Aplicar augmentadores en secuencia."""
        if torch.rand(1).item() > self.p:
            return data
        
        for augmenter in self.augmenters:
            data = augmenter(data)
        
        return data


class AugmentationFactory:
    """Factory para augmentadores."""
    
    _augmenters = {
        'gaussian_noise': GaussianNoiseAugmenter,
        'random_scale': RandomScaleAugmenter,
        'random_shift': RandomShiftAugmenter,
        'random_rotation': RandomRotationAugmenter,
        'time_warp': TimeWarpAugmenter
    }
    
    @classmethod
    def create_augmenter(cls, augment_type: str, **kwargs) -> Augmenter:
        """
        Crear augmentador.
        
        Args:
            augment_type: Tipo de augmentación
            **kwargs: Argumentos adicionales
            
        Returns:
            Augmentador
        """
        if augment_type not in cls._augmenters:
            raise ValueError(f"Unknown augmentation type: {augment_type}")
        
        return cls._augmenters[augment_type](**kwargs)
    
    @classmethod
    def create_compose(cls, augment_types: List[str], **kwargs) -> ComposeAugmenter:
        """
        Crear composición de augmentadores.
        
        Args:
            augment_types: Lista de tipos
            **kwargs: Argumentos adicionales
            
        Returns:
            ComposeAugmenter
        """
        augmenters = [cls.create_augmenter(t) for t in augment_types]
        return ComposeAugmenter(augmenters, **kwargs)
    
    @classmethod
    def register_augmenter(cls, augment_type: str, augmenter_class: type):
        """Registrar nuevo augmentador."""
        cls._augmenters[augment_type] = augmenter_class








