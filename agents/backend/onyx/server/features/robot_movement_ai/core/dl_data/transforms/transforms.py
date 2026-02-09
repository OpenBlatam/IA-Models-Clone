"""
Data Transforms - Modular Data Preprocessing
============================================

Transformaciones modulares para preprocesamiento de datos.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
import torch
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Transform(ABC):
    """Clase base para transformaciones."""
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Aplicar transformación."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """Componer múltiples transformaciones."""
    
    def __init__(self, transforms: List[Transform]):
        """
        Inicializar composición.
        
        Args:
            transforms: Lista de transformaciones
        """
        self.transforms = transforms
    
    def __call__(self, data: Any) -> Any:
        """Aplicar todas las transformaciones en secuencia."""
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def __repr__(self) -> str:
        return f"Compose({[str(t) for t in self.transforms]})"


class Normalize(Transform):
    """Normalizar datos."""
    
    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        axis: int = 0
    ):
        """
        Inicializar normalización.
        
        Args:
            mean: Media (si None, se calcula)
            std: Desviación estándar (si None, se calcula)
            axis: Eje para calcular estadísticas
        """
        self.mean = mean
        self.std = std
        self.axis = axis
        self._fitted = mean is not None and std is not None
    
    def fit(self, data: np.ndarray):
        """Ajustar parámetros de normalización."""
        self.mean = np.mean(data, axis=self.axis, keepdims=True)
        self.std = np.std(data, axis=self.axis, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)  # Evitar división por cero
        self._fitted = True
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Aplicar normalización."""
        if not self._fitted:
            self.fit(data)
        
        return (data - self.mean) / self.std
    
    def inverse(self, data: np.ndarray) -> np.ndarray:
        """Aplicar transformación inversa."""
        if not self._fitted:
            raise ValueError("Transform must be fitted before inverse")
        
        return data * self.std + self.mean


class ToTensor(Transform):
    """Convertir a tensor de PyTorch."""
    
    def __init__(self, dtype: torch.dtype = torch.float32):
        """
        Inicializar conversión.
        
        Args:
            dtype: Tipo de datos del tensor
        """
        self.dtype = dtype
    
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        """Convertir a tensor."""
        if isinstance(data, torch.Tensor):
            return data.to(self.dtype)
        return torch.from_numpy(data).to(self.dtype)


class AddNoise(Transform):
    """Agregar ruido gaussiano."""
    
    def __init__(self, std: float = 0.01):
        """
        Inicializar transformación.
        
        Args:
            std: Desviación estándar del ruido
        """
        self.std = std
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Agregar ruido."""
        noise = np.random.normal(0, self.std, data.shape).astype(data.dtype)
        return data + noise


class RandomScale(Transform):
    """Escalado aleatorio."""
    
    def __init__(self, scale_range: tuple = (0.95, 1.05)):
        """
        Inicializar transformación.
        
        Args:
            scale_range: Rango de escalado (min, max)
        """
        self.scale_range = scale_range
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Aplicar escalado aleatorio."""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return data * scale


class RandomShift(Transform):
    """Desplazamiento aleatorio."""
    
    def __init__(self, shift_range: tuple = (-0.05, 0.05)):
        """
        Inicializar transformación.
        
        Args:
            shift_range: Rango de desplazamiento (min, max)
        """
        self.shift_range = shift_range
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Aplicar desplazamiento aleatorio."""
        shift = np.random.uniform(
            self.shift_range[0],
            self.shift_range[1],
            size=(1, data.shape[-1])
        )
        return data + shift


class PadSequence(Transform):
    """Padding de secuencias."""
    
    def __init__(self, max_length: int, pad_value: float = 0.0):
        """
        Inicializar transformación.
        
        Args:
            max_length: Longitud máxima
            pad_value: Valor de padding
        """
        self.max_length = max_length
        self.pad_value = pad_value
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Aplicar padding."""
        if len(data) >= self.max_length:
            return data[:self.max_length]
        
        pad_length = self.max_length - len(data)
        pad = np.full((pad_length, *data.shape[1:]), self.pad_value, dtype=data.dtype)
        return np.concatenate([data, pad], axis=0)


class TruncateSequence(Transform):
    """Truncar secuencias."""
    
    def __init__(self, max_length: int):
        """
        Inicializar transformación.
        
        Args:
            max_length: Longitud máxima
        """
        self.max_length = max_length
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Aplicar truncamiento."""
        if len(data) <= self.max_length:
            return data
        
        start_idx = np.random.randint(0, len(data) - self.max_length + 1)
        return data[start_idx:start_idx + self.max_length]


def create_training_transforms(
    normalize: bool = True,
    augment: bool = False,
    max_length: Optional[int] = None
) -> Compose:
    """
    Crear transformaciones para entrenamiento.
    
    Args:
        normalize: Aplicar normalización
        augment: Aplicar data augmentation
        max_length: Longitud máxima de secuencias
        
    Returns:
        Composición de transformaciones
    """
    transforms = []
    
    if max_length:
        transforms.append(PadSequence(max_length))
    
    if normalize:
        transforms.append(Normalize())
    
    if augment:
        transforms.extend([
            AddNoise(std=0.01),
            RandomScale(scale_range=(0.95, 1.05)),
            RandomShift(shift_range=(-0.05, 0.05))
        ])
    
    transforms.append(ToTensor())
    
    return Compose(transforms)


def create_validation_transforms(
    normalize: bool = True,
    max_length: Optional[int] = None
) -> Compose:
    """
    Crear transformaciones para validación.
    
    Args:
        normalize: Aplicar normalización
        max_length: Longitud máxima de secuencias
        
    Returns:
        Composición de transformaciones
    """
    transforms = []
    
    if max_length:
        transforms.append(PadSequence(max_length))
    
    if normalize:
        transforms.append(Normalize())
    
    transforms.append(ToTensor())
    
    return Compose(transforms)













