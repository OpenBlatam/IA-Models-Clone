"""
Routing Data Loader
===================

DataLoader profesional para datasets de routing con optimizaciones avanzadas.
Implementa prefetching, caching, y transformaciones eficientes.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler
    from torch.utils.data.distributed import DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. DataLoader features will be disabled.")


@dataclass
class RouteDataPoint:
    """Punto de datos para entrenamiento de routing."""
    node_features: np.ndarray
    edge_features: np.ndarray
    graph_structure: Dict[str, Any]
    target_route: List[str]
    target_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class RouteDataset(Dataset):
    """Dataset profesional para routing."""
    
    def __init__(
        self,
        data_points: List[RouteDataPoint],
        transform: Optional[Callable] = None,
        cache_enabled: bool = True
    ):
        """
        Inicializar dataset.
        
        Args:
            data_points: Lista de puntos de datos
            transform: Transformación a aplicar
            cache_enabled: Habilitar cache de datos procesados
        """
        self.data_points = data_points
        self.transform = transform
        self.cache_enabled = cache_enabled
        self._cache: Dict[int, Dict[str, Any]] = {}
    
    def __len__(self) -> int:
        return len(self.data_points)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Obtener item del dataset."""
        # Verificar cache
        if self.cache_enabled and idx in self._cache:
            return self._cache[idx]
        
        data_point = self.data_points[idx]
        
        # Convertir a tensores
        sample = {
            'node_features': torch.FloatTensor(data_point.node_features),
            'edge_features': torch.FloatTensor(data_point.edge_features),
            'target_route': data_point.target_route,
            'target_metrics': torch.FloatTensor([
                data_point.target_metrics.get('distance', 0.0),
                data_point.target_metrics.get('time', 0.0),
                data_point.target_metrics.get('cost', 0.0)
            ]),
            'metadata': data_point.metadata
        }
        
        # Aplicar transformación
        if self.transform:
            sample = self.transform(sample)
        
        # Guardar en cache
        if self.cache_enabled:
            self._cache[idx] = sample
        
        return sample
    
    def clear_cache(self):
        """Limpiar cache."""
        self._cache.clear()


class FastDataLoader:
    """DataLoader optimizado con prefetching y caching."""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        drop_last: bool = False
    ):
        """
        Inicializar DataLoader optimizado.
        
        Args:
            dataset: Dataset de PyTorch
            batch_size: Tamaño de batch
            shuffle: Mezclar datos
            num_workers: Número de workers para carga
            pin_memory: Pin memory para transferencia rápida a GPU
            prefetch_factor: Factor de prefetching
            persistent_workers: Mantener workers persistentes
            drop_last: Eliminar último batch incompleto
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for FastDataLoader")
        
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Crear DataLoader optimizado
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=drop_last
        )
    
    def __iter__(self):
        """Iterador del DataLoader."""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Longitud del DataLoader."""
        return len(self.dataloader)


class DataAugmentation:
    """Transformaciones de data augmentation para routing."""
    
    @staticmethod
    def add_noise(features: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        """Agregar ruido gaussiano."""
        noise = torch.randn_like(features) * noise_level
        return features + noise
    
    @staticmethod
    def scale_features(features: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """Escalar features aleatoriamente."""
        scale = torch.empty(1).uniform_(*scale_range)
        return features * scale
    
    @staticmethod
    def random_dropout(features: torch.Tensor, dropout_prob: float = 0.1) -> torch.Tensor:
        """Aplicar dropout aleatorio."""
        mask = torch.rand_like(features) > dropout_prob
        return features * mask
    
    @staticmethod
    def compose_transforms(transforms: List[Callable]) -> Callable:
        """Componer múltiples transformaciones."""
        def composed(sample: Dict[str, Any]) -> Dict[str, Any]:
            for transform in transforms:
                sample = transform(sample)
            return sample
        return composed


class BalancedSampler(Sampler):
    """Sampler balanceado para datasets desbalanceados."""
    
    def __init__(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        replacement: bool = True
    ):
        """
        Inicializar sampler balanceado.
        
        Args:
            dataset: Dataset
            num_samples: Número de samples (usa len(dataset) si None)
            replacement: Muestreo con reemplazo
        """
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.replacement = replacement
    
    def __iter__(self):
        """Generar índices balanceados."""
        indices = torch.randint(
            high=len(self.dataset),
            size=(self.num_samples,),
            dtype=torch.int64
        ).tolist()
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples

