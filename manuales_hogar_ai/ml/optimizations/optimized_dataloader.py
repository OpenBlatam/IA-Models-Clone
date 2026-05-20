"""
Optimized DataLoader
====================

DataLoader optimizado para máximo rendimiento.
"""

import logging
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Any, Callable
import multiprocessing

logger = logging.getLogger(__name__)


class OptimizedDataLoader:
    """DataLoader optimizado."""
    
    @staticmethod
    def create_optimized_loader(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ) -> DataLoader:
        """
        Crear DataLoader optimizado.
        
        Args:
            dataset: Dataset
            batch_size: Tamaño de batch
            shuffle: Mezclar datos
            num_workers: Número de workers (None = auto)
            pin_memory: Pin memory para GPU
            prefetch_factor: Factor de prefetch
            persistent_workers: Workers persistentes
        
        Returns:
            DataLoader optimizado
        """
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=False
        )
        
        logger.info(
            f"DataLoader optimizado creado: "
            f"batch_size={batch_size}, workers={num_workers}, "
            f"pin_memory={pin_memory}, prefetch={prefetch_factor}"
        )
        
        return loader
    
    @staticmethod
    def create_fast_loader(
        dataset: Dataset,
        batch_size: int = 64,
        **kwargs
    ) -> DataLoader:
        """
        Crear DataLoader ultra-rápido.
        
        Args:
            dataset: Dataset
            batch_size: Tamaño de batch
            **kwargs: Otros parámetros
        
        Returns:
            DataLoader rápido
        """
        return OptimizedDataLoader.create_optimized_loader(
            dataset,
            batch_size=batch_size,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            **kwargs
        )




