"""
Pipeline Utils - Utilidades de Pipelines de Datos
==================================================

Utilidades para crear pipelines de datos avanzados.
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Callable, Optional, Dict, Any, Iterator
import numpy as np
from functools import partial

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Pipeline de datos con transformaciones encadenadas.
    """
    
    def __init__(self):
        """Inicializar pipeline."""
        self.transforms: List[Callable] = []
        self.target_transforms: List[Callable] = []
    
    def add_transform(self, transform: Callable) -> 'DataPipeline':
        """
        Agregar transformación a datos.
        
        Args:
            transform: Función de transformación
            
        Returns:
            Self para chaining
        """
        self.transforms.append(transform)
        return self
    
    def add_target_transform(self, transform: Callable) -> 'DataPipeline':
        """
        Agregar transformación a targets.
        
        Args:
            transform: Función de transformación
            
        Returns:
            Self para chaining
        """
        self.target_transforms.append(transform)
        return self
    
    def apply(self, data: Any, target: Optional[Any] = None) -> tuple:
        """
        Aplicar pipeline.
        
        Args:
            data: Datos
            target: Target opcional
            
        Returns:
            Tupla (datos transformados, target transformado)
        """
        for transform in self.transforms:
            data = transform(data)
        
        if target is not None:
            for transform in self.target_transforms:
                target = transform(target)
        
        return data, target
    
    def __call__(self, data: Any, target: Optional[Any] = None) -> tuple:
        """Llamar pipeline."""
        return self.apply(data, target)


class ParallelDataLoader:
    """
    DataLoader con procesamiento paralelo.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        **kwargs
    ):
        """
        Inicializar parallel data loader.
        
        Args:
            dataset: Dataset
            batch_size: Tamaño de batch
            num_workers: Número de workers
            pin_memory: Pin memory
            prefetch_factor: Factor de prefetch
            **kwargs: Argumentos adicionales
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.kwargs = kwargs
    
    def get_loader(self) -> DataLoader:
        """
        Obtener DataLoader.
        
        Returns:
            DataLoader configurado
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            **self.kwargs
        )


class StreamingDataset(IterableDataset):
    """
    Dataset para streaming de datos.
    """
    
    def __init__(
        self,
        data_source: Iterator,
        transform: Optional[Callable] = None
    ):
        """
        Inicializar streaming dataset.
        
        Args:
            data_source: Fuente de datos (iterador)
            transform: Transformación opcional
        """
        self.data_source = data_source
        self.transform = transform
    
    def __iter__(self) -> Iterator:
        """
        Iterar sobre datos.
        
        Returns:
            Iterador
        """
        for item in self.data_source:
            if self.transform:
                item = self.transform(item)
            yield item


class CachedDataset(Dataset):
    """
    Dataset con caché en memoria.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        cache_size: Optional[int] = None
    ):
        """
        Inicializar cached dataset.
        
        Args:
            dataset: Dataset base
            cache_size: Tamaño de caché (None = ilimitado)
        """
        self.dataset = dataset
        self.cache_size = cache_size
        self.cache: Dict[int, Any] = {}
    
    def __len__(self) -> int:
        """Longitud del dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Any:
        """
        Obtener item.
        
        Args:
            idx: Índice
            
        Returns:
            Item
        """
        if idx in self.cache:
            return self.cache[idx]
        
        item = self.dataset[idx]
        
        if self.cache_size is None or len(self.cache) < self.cache_size:
            self.cache[idx] = item
        
        return item
    
    def clear_cache(self):
        """Limpiar caché."""
        self.cache.clear()


class BatchProcessor:
    """
    Procesador de batches con múltiples operaciones.
    """
    
    def __init__(self):
        """Inicializar procesador."""
        self.processors: List[Callable] = []
    
    def add_processor(self, processor: Callable) -> 'BatchProcessor':
        """
        Agregar procesador.
        
        Args:
            processor: Función procesadora
            
        Returns:
            Self para chaining
        """
        self.processors.append(processor)
        return self
    
    def process(self, batch: Any) -> Any:
        """
        Procesar batch.
        
        Args:
            batch: Batch a procesar
            
        Returns:
            Batch procesado
        """
        for processor in self.processors:
            batch = processor(batch)
        return batch
    
    def __call__(self, batch: Any) -> Any:
        """Llamar procesador."""
        return self.process(batch)


class DataPrefetcher:
    """
    Prefetcher de datos para GPU.
    """
    
    def __init__(self, loader: DataLoader, device: str = "cuda"):
        """
        Inicializar prefetcher.
        
        Args:
            loader: DataLoader
            device: Dispositivo
        """
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream() if device == "cuda" else None
        self.next_batch = None
        self._preload()
    
    def _preload(self):
        """Precargar siguiente batch."""
        try:
            self.next_batch = next(self.loader)
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    self.next_batch = self._move_to_device(self.next_batch)
        except StopIteration:
            self.next_batch = None
    
    def _move_to_device(self, batch: Any) -> Any:
        """
        Mover batch a dispositivo.
        
        Args:
            batch: Batch
            
        Returns:
            Batch en dispositivo
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(item) for item in batch)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        else:
            return batch
    
    def __iter__(self) -> Iterator:
        """Iterar sobre batches."""
        return self
    
    def __next__(self) -> Any:
        """
        Obtener siguiente batch.
        
        Returns:
            Batch
        """
        if self.next_batch is None:
            raise StopIteration
        
        batch = self.next_batch
        self._preload()
        
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        return batch


class DataBalancer:
    """
    Balanceador de datos para clases desbalanceadas.
    """
    
    def __init__(self, strategy: str = "oversample"):
        """
        Inicializar balanceador.
        
        Args:
            strategy: Estrategia ('oversample', 'undersample', 'smote')
        """
        self.strategy = strategy
    
    def balance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple:
        """
        Balancear datos.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tupla (X_balanced, y_balanced)
        """
        if self.strategy == "oversample":
            return self._oversample(X, y)
        elif self.strategy == "undersample":
            return self._undersample(X, y)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _oversample(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Oversampling de clase minoritaria."""
        from collections import Counter
        
        class_counts = Counter(y)
        max_count = max(class_counts.values())
        
        X_balanced = [X]
        y_balanced = [y]
        
        for class_label, count in class_counts.items():
            if count < max_count:
                indices = np.where(y == class_label)[0]
                num_samples = max_count - count
                oversampled_indices = np.random.choice(indices, num_samples, replace=True)
                X_balanced.append(X[oversampled_indices])
                y_balanced.append(y[oversampled_indices])
        
        return np.concatenate(X_balanced), np.concatenate(y_balanced)
    
    def _undersample(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Undersampling de clase mayoritaria."""
        from collections import Counter
        
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        
        X_balanced = []
        y_balanced = []
        
        for class_label in class_counts.keys():
            indices = np.where(y == class_label)[0]
            sampled_indices = np.random.choice(indices, min_count, replace=False)
            X_balanced.append(X[sampled_indices])
            y_balanced.append(y[sampled_indices])
        
        return np.concatenate(X_balanced), np.concatenate(y_balanced)




