"""
Dataset Utils - Utilidades para datasets
========================================

Utilidades para crear y gestionar datasets personalizados.
Sigue mejores prácticas de PyTorch datasets.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class BaseDataset(Dataset, ABC):
    """Clase base para datasets personalizados"""
    
    def __init__(self, transform: Optional[Callable] = None):
        """
        Args:
            transform: Transformación a aplicar
        """
        self.transform = transform
    
    @abstractmethod
    def __len__(self) -> int:
        """Retornar tamaño del dataset"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Obtener item del dataset"""
        pass
    
    def apply_transform(self, item: Any) -> Any:
        """Aplicar transformación si existe"""
        if self.transform:
            return self.transform(item)
        return item


class TensorDataset(BaseDataset):
    """Dataset simple de tensores"""
    
    def __init__(
        self,
        data: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            data: Tensor de datos
            targets: Tensor de targets (opcional)
            transform: Transformación (opcional)
        """
        super().__init__(transform)
        self.data = data
        self.targets = targets
        
        if targets is not None and len(data) != len(targets):
            raise ValueError("Data and targets must have same length")
    
    def __len__(self) -> int:
        """Tamaño del dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Obtener item"""
        data_item = self.data[idx]
        data_item = self.apply_transform(data_item)
        
        if self.targets is not None:
            return data_item, self.targets[idx]
        return data_item, None


class DatasetUtils:
    """Utilidades para datasets"""
    
    @staticmethod
    def concatenate_datasets(datasets: List[Dataset]) -> ConcatDataset:
        """
        Concatenar múltiples datasets.
        
        Args:
            datasets: Lista de datasets
        
        Returns:
            Dataset concatenado
        """
        if not datasets:
            raise ValueError("Cannot concatenate empty list of datasets")
        
        return ConcatDataset(datasets)
    
    @staticmethod
    def split_dataset(
        dataset: Dataset,
        lengths: List[int],
        generator: Optional[torch.Generator] = None
    ) -> List[Subset]:
        """
        Dividir dataset en múltiples subsets.
        
        Args:
            dataset: Dataset a dividir
            lengths: Lista de longitudes para cada subset
            generator: Generator para reproducibilidad
        
        Returns:
            Lista de subsets
        """
        from torch.utils.data import random_split
        
        if sum(lengths) != len(dataset):
            raise ValueError(f"Sum of lengths ({sum(lengths)}) must equal dataset length ({len(dataset)})")
        
        return random_split(dataset, lengths, generator=generator)
    
    @staticmethod
    def filter_dataset(
        dataset: Dataset,
        filter_fn: Callable[[int], bool]
    ) -> List[int]:
        """
        Filtrar dataset y retornar índices que cumplen condición.
        
        Args:
            dataset: Dataset
            filter_fn: Función de filtrado (idx -> bool)
        
        Returns:
            Lista de índices que cumplen condición
        """
        indices = []
        for idx in range(len(dataset)):
            if filter_fn(idx):
                indices.append(idx)
        return indices
    
    @staticmethod
    def get_dataset_statistics(
        dataset: Dataset,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Obtener estadísticas del dataset.
        
        Args:
            dataset: Dataset
            sample_size: Tamaño de muestra (None = todo el dataset)
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "total_size": len(dataset),
            "sampled_size": 0,
        }
        
        if sample_size is None:
            sample_size = len(dataset)
        else:
            sample_size = min(sample_size, len(dataset))
        
        try:
            # Sample items
            sample_indices = torch.randperm(len(dataset))[:sample_size]
            
            data_samples = []
            target_samples = []
            
            for idx in sample_indices:
                item = dataset[int(idx)]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    data_samples.append(item[0])
                    target_samples.append(item[1])
                else:
                    data_samples.append(item)
            
            # Calculate statistics
            if data_samples:
                if isinstance(data_samples[0], torch.Tensor):
                    stacked = torch.stack(data_samples)
                    stats.update({
                        "data_shape": list(stacked.shape[1:]),
                        "data_mean": float(stacked.mean().item()),
                        "data_std": float(stacked.std().item()),
                        "data_min": float(stacked.min().item()),
                        "data_max": float(stacked.max().item()),
                    })
            
            if target_samples:
                if isinstance(target_samples[0], torch.Tensor):
                    stacked = torch.stack(target_samples)
                    stats.update({
                        "target_shape": list(stacked.shape[1:]),
                        "target_mean": float(stacked.mean().item()),
                        "target_std": float(stacked.std().item()),
                    })
                    
                    # Class distribution if classification
                    if stacked.dim() == 1:
                        unique, counts = torch.unique(stacked, return_counts=True)
                        stats["class_distribution"] = {
                            int(k.item()): int(v.item())
                            for k, v in zip(unique, counts)
                        }
            
            stats["sampled_size"] = sample_size
        
        except Exception as e:
            logger.warning(f"Error calculating dataset statistics: {e}")
        
        return stats
    
    @staticmethod
    def create_subset(
        dataset: Dataset,
        indices: List[int]
    ) -> Subset:
        """
        Crear subset del dataset.
        
        Args:
            dataset: Dataset
            indices: Lista de índices
        
        Returns:
            Subset
        """
        return Subset(dataset, indices)




