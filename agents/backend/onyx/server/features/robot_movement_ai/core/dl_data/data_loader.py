"""
Data Loader Utilities
======================

Utilidades para crear DataLoaders.
"""

import logging
from typing import Optional, Tuple
import numpy as np

try:
    from torch.utils.data import DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    DataLoader = None
    random_split = None

from .datasets import RobotDataset, RobotSequenceDataset
from .data_transforms import DataTransform

logger = logging.getLogger(__name__)


def create_dataloader(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    transform: Optional[DataTransform] = None,
    num_workers: int = 0,
    pin_memory: bool = False
):
    """
    Crear DataLoader básico.
    
    Args:
        inputs: Datos de entrada
        targets: Datos objetivo
        batch_size: Tamaño de batch
        shuffle: Mezclar datos
        transform: Transformación (opcional)
        num_workers: Número de workers
        pin_memory: Pin memory para GPU
        
    Returns:
        DataLoader
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DataLoader")
    
    dataset = RobotDataset(inputs, targets, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def create_train_val_loaders(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    train_transform: Optional[DataTransform] = None,
    val_transform: Optional[DataTransform] = None,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Crear DataLoaders de entrenamiento y validación.
    
    Args:
        inputs: Datos de entrada
        targets: Datos objetivo
        batch_size: Tamaño de batch
        val_split: Proporción de validación
        train_transform: Transformación para entrenamiento
        val_transform: Transformación para validación
        num_workers: Número de workers
        pin_memory: Pin memory para GPU
        
    Returns:
        Tupla (train_loader, val_loader)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DataLoader")
    
    dataset = RobotDataset(inputs, targets)
    
    # Split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Aplicar transformaciones
    if train_transform:
        train_dataset.dataset.transform = train_transform
    if val_transform:
        val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader




