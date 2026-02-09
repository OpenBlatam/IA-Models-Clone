"""
Pipeline Datasets Module
========================

Datasets profesionales para entrenamiento de modelos de deep learning.
Incluye transformaciones, augmentaciones y manejo eficiente de datos.
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from abc import ABC, abstractmethod
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Dataset = None
    DataLoader = None
    logging.warning("PyTorch not available. Datasets will be disabled.")

logger = logging.getLogger(__name__)


class BaseTrajectoryDataset(Dataset, ABC):
    """
    Clase base abstracta para datasets de trayectorias.
    
    Proporciona interfaz común para diferentes tipos de datasets.
    """
    
    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Inicializar dataset base.
        
        Args:
            transform: Transformación a aplicar a inputs
            target_transform: Transformación a aplicar a targets
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for BaseTrajectoryDataset")
        
        self.transform = transform
        self.target_transform = target_transform
    
    @abstractmethod
    def __len__(self) -> int:
        """Retornar tamaño del dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Obtener item del dataset."""
        pass


class TrajectoryDataset(BaseTrajectoryDataset):
    """
    Dataset para trayectorias de robot.
    
    Soporta diferentes formatos de entrada y transformaciones.
    """
    
    def __init__(
        self,
        trajectories: np.ndarray,
        targets: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        normalize: bool = True,
        cache: bool = False
    ):
        """
        Inicializar dataset de trayectorias.
        
        Args:
            trajectories: Array de trayectorias (N, seq_len, features)
            targets: Array de targets (N, seq_len, output_dim) o (N, output_dim)
            transform: Transformación a aplicar a inputs
            target_transform: Transformación a aplicar a targets
            normalize: Normalizar datos a [0, 1]
            cache: Cachear datos en memoria
        """
        super().__init__(transform, target_transform)
        
        if trajectories.ndim < 2:
            raise ValueError(f"Trajectories must be at least 2D, got {trajectories.ndim}D")
        
        # Normalizar si se solicita
        if normalize:
            self.mean = trajectories.mean(axis=(0, 1), keepdims=True)
            self.std = trajectories.std(axis=(0, 1), keepdims=True) + 1e-8
            trajectories = (trajectories - self.mean) / self.std
            self.normalize = True
        else:
            self.mean = None
            self.std = None
            self.normalize = False
        
        # Convertir a tensores
        if cache:
            self.trajectories = torch.FloatTensor(trajectories)
            self.targets = torch.FloatTensor(targets) if targets is not None else None
        else:
            self.trajectories = trajectories.astype(np.float32)
            self.targets = targets.astype(np.float32) if targets is not None else None
        
        self.cache = cache
        logger.info(f"TrajectoryDataset initialized: {len(self)} samples, cache={cache}")
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Obtener item del dataset."""
        if self.cache:
            trajectory = self.trajectories[idx]
        else:
            trajectory = torch.FloatTensor(self.trajectories[idx])
        
        if self.transform:
            trajectory = self.transform(trajectory)
        
        result = {"input": trajectory}
        
        if self.targets is not None:
            if self.cache:
                target = self.targets[idx]
            else:
                target = torch.FloatTensor(self.targets[idx])
            
            if self.target_transform:
                target = self.target_transform(target)
            
            result["target"] = target
        
        return result
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Desnormalizar datos."""
        if not self.normalize or self.mean is None or self.std is None:
            return data
        
        if isinstance(data, torch.Tensor):
            mean = torch.FloatTensor(self.mean.squeeze())
            std = torch.FloatTensor(self.std.squeeze())
            return data * std + mean
        else:
            return data * self.std + self.mean


class SequenceDataset(BaseTrajectoryDataset):
    """
    Dataset para secuencias temporales.
    
    Útil para modelos que requieren contexto temporal.
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: Optional[np.ndarray] = None,
        sequence_length: int = 10,
        stride: int = 1,
        transform: Optional[Callable] = None
    ):
        """
        Inicializar dataset de secuencias.
        
        Args:
            sequences: Array de secuencias (N, time_steps, features)
            targets: Array de targets (N, output_dim)
            sequence_length: Longitud de secuencia a usar
            stride: Stride para crear secuencias
            transform: Transformación a aplicar
        """
        super().__init__(transform)
        
        self.sequences = sequences
        self.targets = targets
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Crear índices de secuencias
        self.indices = []
        for i in range(0, len(sequences) - sequence_length + 1, stride):
            self.indices.append(i)
        
        logger.info(f"SequenceDataset initialized: {len(self)} sequences")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Obtener secuencia del dataset."""
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        sequence = torch.FloatTensor(self.sequences[start_idx:end_idx])
        
        if self.transform:
            sequence = self.transform(sequence)
        
        result = {"input": sequence}
        
        if self.targets is not None:
            target = torch.FloatTensor(self.targets[start_idx])
            result["target"] = target
        
        return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
) -> DataLoader:
    """
    Crear DataLoader optimizado.
    
    Args:
        dataset: Dataset de PyTorch
        batch_size: Tamaño de batch
        shuffle: Mezclar datos
        num_workers: Número de workers
        pin_memory: Pin memory para GPU
        drop_last: Descartar último batch incompleto
        prefetch_factor: Factor de prefetch
        persistent_workers: Mantener workers vivos entre épocas
        
    Returns:
        DataLoader configurado
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for create_dataloader")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )

