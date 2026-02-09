"""
Data Loading Optimization
=========================

Optimizaciones para carga de datos más rápida.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Any, Callable
import numpy as np
from collections import deque
import threading
import queue
import logging

logger = logging.getLogger(__name__)


class FastDataLoader(DataLoader):
    """
    DataLoader optimizado para velocidad.
    """
    
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = True,
                 **kwargs):
        """
        Inicializar DataLoader rápido.
        
        Args:
            dataset: Dataset
            batch_size: Tamaño de batch
            shuffle: Mezclar
            num_workers: Número de workers
            pin_memory: Pin memory para GPU
            prefetch_factor: Factor de prefetch
            persistent_workers: Mantener workers vivos
        """
        super(FastDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            **kwargs
        )


class CachedDataset(Dataset):
    """
    Dataset con cache para acceso rápido.
    """
    
    def __init__(self, dataset: Dataset, cache_size: int = 1000):
        """
        Inicializar dataset con cache.
        
        Args:
            dataset: Dataset original
            cache_size: Tamaño del cache
        """
        self.dataset = dataset
        self.cache = {}
        self.cache_size = cache_size
        self.access_order = deque()
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        """Obtener item con cache."""
        if idx in self.cache:
            # Mover al final (LRU)
            self.access_order.remove(idx)
            self.access_order.append(idx)
            return self.cache[idx]
        
        # Obtener del dataset
        item = self.dataset[idx]
        
        # Agregar al cache
        if len(self.cache) >= self.cache_size:
            # Remover LRU
            lru_idx = self.access_order.popleft()
            del self.cache[lru_idx]
        
        self.cache[idx] = item
        self.access_order.append(idx)
        
        return item
    
    def clear_cache(self):
        """Limpiar cache."""
        self.cache.clear()
        self.access_order.clear()


class PrefetchDataLoader:
    """
    DataLoader con prefetching asíncrono.
    """
    
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 prefetch_batches: int = 3):
        """
        Inicializar prefetch loader.
        
        Args:
            dataset: Dataset
            batch_size: Tamaño de batch
            shuffle: Mezclar
            num_workers: Número de workers
            prefetch_batches: Número de batches a prefetchear
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        
        self.queue = queue.Queue(maxsize=prefetch_batches)
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Crear índices
        self.indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
    
    def _worker(self):
        """Worker thread para prefetching."""
        while not self.stop_event.is_set():
            try:
                batch_indices = []
                for _ in range(self.batch_size):
                    if self.current_idx >= len(self.indices):
                        if self.shuffle:
                            np.random.shuffle(self.indices)
                        self.current_idx = 0
                    
                    batch_indices.append(self.indices[self.current_idx])
                    self.current_idx += 1
                
                batch = [self.dataset[idx] for idx in batch_indices]
                
                # Stack tensors
                features = torch.stack([b[0] for b in batch])
                targets = torch.stack([b[1] for b in batch])
                metadata = [b[2] for b in batch]
                
                self.queue.put((features, targets, metadata), timeout=1)
            except queue.Full:
                continue
            except Exception as e:
                logger.error(f"Error en prefetch worker: {e}")
                break
    
    def __iter__(self):
        """Iterador."""
        self.stop_event.clear()
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        while True:
            try:
                batch = self.queue.get(timeout=5)
                yield batch
            except queue.Empty:
                if not self.worker_thread.is_alive():
                    break
                continue
    
    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
    
    def stop(self):
        """Detener worker."""
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join()


class OptimizedDataPipeline:
    """
    Pipeline de datos optimizado.
    """
    
    @staticmethod
    def create_fast_loader(dataset: Dataset,
                          batch_size: int = 32,
                          use_cache: bool = True,
                          use_prefetch: bool = False,
                          num_workers: int = 4) -> DataLoader:
        """
        Crear DataLoader optimizado.
        
        Args:
            dataset: Dataset
            batch_size: Tamaño de batch
            use_cache: Usar cache
            use_prefetch: Usar prefetch asíncrono
            num_workers: Número de workers
            
        Returns:
            DataLoader optimizado
        """
        # Aplicar cache si se solicita
        if use_cache:
            dataset = CachedDataset(dataset, cache_size=1000)
        
        # Usar prefetch asíncrono o DataLoader estándar optimizado
        if use_prefetch:
            return PrefetchDataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )
        else:
            return FastDataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

