"""
Advanced Data Pipelines - Pipelines de datos avanzados
========================================================
Prefetching, caching, y optimización de pipelines
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Any, Optional, Callable, Iterator
from collections import deque
import threading
import queue
import time

logger = logging.getLogger(__name__)


class PrefetchDataLoader:
    """DataLoader con prefetching"""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


class CachedDataset(Dataset):
    """Dataset con caching"""
    
    def __init__(
        self,
        base_dataset: Dataset,
        cache_size: int = 1000,
        cache_key_fn: Optional[Callable] = None
    ):
        self.base_dataset = base_dataset
        self.cache_size = cache_size
        self.cache_key_fn = cache_key_fn or (lambda idx: str(idx))
        self.cache: Dict[str, Any] = {}
        self.cache_order: deque = deque(maxlen=cache_size)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        cache_key = self.cache_key_fn(idx)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Cargar y cachear
        item = self.base_dataset[idx]
        
        if len(self.cache) >= self.cache_size:
            # Eliminar más antiguo
            oldest_key = self.cache_order[0]
            del self.cache[oldest_key]
            self.cache_order.popleft()
        
        self.cache[cache_key] = item
        self.cache_order.append(cache_key)
        
        return item
    
    def clear_cache(self):
        """Limpia cache"""
        self.cache.clear()
        self.cache_order.clear()


class AsyncDataLoader:
    """DataLoader asíncrono con prefetching en background"""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        queue_size: int = 10
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue_size = queue_size
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2
        )
        
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = None
        self.stop_event = threading.Event()
    
    def _prefetch_worker(self):
        """Worker thread para prefetching"""
        try:
            for batch in self.dataloader:
                if self.stop_event.is_set():
                    break
                self.queue.put(batch)
        except Exception as e:
            logger.error(f"Prefetch worker error: {e}")
        finally:
            self.queue.put(None)  # Sentinel
    
    def __iter__(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()
        
        while True:
            batch = self.queue.get()
            if batch is None:
                break
            yield batch
    
    def __len__(self):
        return len(self.dataloader)
    
    def stop(self):
        """Detiene prefetching"""
        self.stop_event.set()
        if self.thread:
            self.thread.join()


class DataPipeline:
    """Pipeline de datos con múltiples transformaciones"""
    
    def __init__(self):
        self.transforms: List[Callable] = []
    
    def add_transform(self, transform: Callable):
        """Agrega transformación"""
        self.transforms.append(transform)
        return self
    
    def __call__(self, data: Any) -> Any:
        """Aplica todas las transformaciones"""
        result = data
        for transform in self.transforms:
            result = transform(result)
        return result


class BalancedSampler:
    """Sampler balanceado para clases"""
    
    def __init__(self, dataset: Dataset, num_samples: Optional[int] = None):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        
        # Asumir que dataset tiene atributo targets o labels
        if hasattr(dataset, "targets"):
            self.targets = dataset.targets
        elif hasattr(dataset, "labels"):
            self.targets = dataset.labels
        else:
            # Fallback: muestrear uniformemente
            self.targets = None
    
    def __iter__(self) -> Iterator[int]:
        if self.targets is None:
            # Muestreo uniforme
            indices = torch.randperm(len(self.dataset))[:self.num_samples].tolist()
        else:
            # Muestreo balanceado
            from collections import Counter
            class_counts = Counter(self.targets)
            samples_per_class = self.num_samples // len(class_counts)
            
            indices = []
            for class_idx in class_counts.keys():
                class_indices = [i for i, t in enumerate(self.targets) if t == class_idx]
                sampled = torch.randperm(len(class_indices))[:samples_per_class].tolist()
                indices.extend([class_indices[i] for i in sampled])
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples




