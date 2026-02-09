"""
Data Transforms
===============

Transformaciones para datos de robots.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class DataTransform(ABC):
    """Clase base para transformaciones de datos."""
    
    @abstractmethod
    def __call__(self, data):
        """Aplicar transformación."""
        pass


class NormalizeTransform(DataTransform):
    """
    Normalización de datos.
    
    Normaliza datos usando media y desviación estándar.
    """
    
    def __init__(
        self,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        fit_data: Optional[np.ndarray] = None
    ):
        """
        Inicializar normalización.
        
        Args:
            mean: Media (si None, se calcula de fit_data)
            std: Desviación estándar (si None, se calcula de fit_data)
            fit_data: Datos para calcular mean/std
        """
        if fit_data is not None:
            self.mean = float(np.mean(fit_data))
            self.std = float(np.std(fit_data))
            if self.std == 0:
                self.std = 1.0
        else:
            self.mean = mean if mean is not None else 0.0
            self.std = std if std is not None else 1.0
    
    def __call__(self, data):
        """Normalizar datos."""
        if isinstance(data, torch.Tensor):
            return (data - self.mean) / self.std
        else:
            return (data - self.mean) / self.std


class AugmentTransform(DataTransform):
    """
    Transformación de aumento de datos.
    
    Aplica aumentos aleatorios a los datos.
    """
    
    def __init__(
        self,
        noise_std: float = 0.01,
        scale_range: tuple = (0.95, 1.05),
        apply_probability: float = 0.5
    ):
        """
        Inicializar aumento.
        
        Args:
            noise_std: Desviación estándar del ruido
            scale_range: Rango de escala
            apply_probability: Probabilidad de aplicar aumento
        """
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.apply_probability = apply_probability
    
    def __call__(self, data):
        """Aplicar aumento."""
        import random
        
        if random.random() > self.apply_probability:
            return data
        
        if isinstance(data, torch.Tensor):
            augmented = data.clone()
            
            # Agregar ruido
            if self.noise_std > 0:
                noise = torch.randn_like(augmented) * self.noise_std
                augmented = augmented + noise
            
            # Escalar
            scale = random.uniform(*self.scale_range)
            augmented = augmented * scale
            
            return augmented
        else:
            augmented = data.copy()
            
            # Agregar ruido
            if self.noise_std > 0:
                noise = np.random.randn(*augmented.shape) * self.noise_std
                augmented = augmented + noise
            
            # Escalar
            scale = random.uniform(*self.scale_range)
            augmented = augmented * scale
            
            return augmented


class ComposeTransform(DataTransform):
    """
    Composición de transformaciones.
    
    Aplica múltiples transformaciones en secuencia.
    """
    
    def __init__(self, transforms: list):
        """
        Inicializar composición.
        
        Args:
            transforms: Lista de transformaciones
        """
        self.transforms = transforms
    
    def __call__(self, data):
        """Aplicar todas las transformaciones."""
        for transform in self.transforms:
            data = transform(data)
        return data




