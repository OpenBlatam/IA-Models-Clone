"""
Prefetch Optimizer - Optimizador de Prefetch
=============================================

Sistema de prefetching avanzado para máxima velocidad.
"""

import logging
import torch
from torch.utils.data import DataLoader
from typing import Optional, Iterator
from collections import deque

logger = logging.getLogger(__name__)


class PrefetchDataLoader:
    """DataLoader con prefetching optimizado"""
    
    def __init__(
        self,
        dataloader: DataLoader,
        device: str = "cuda",
        prefetch_factor: int = 4
    ):
        """
        Inicializar DataLoader con prefetch
        
        Args:
            dataloader: DataLoader original
            device: Dispositivo
            prefetch_factor: Factor de prefetch
        """
        self.dataloader = dataloader
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.prefetch_queue = deque(maxlen=prefetch_factor)
        
        logger.info(f"Prefetch DataLoader inicializado (factor={prefetch_factor})")
    
    def __iter__(self) -> Iterator:
        """Iterador con prefetch"""
        dataloader_iter = iter(self.dataloader)
        
        # Prefetch inicial
        for _ in range(self.prefetch_factor):
            try:
                batch = next(dataloader_iter)
                prefetched = self._prefetch_batch(batch)
                self.prefetch_queue.append(prefetched)
            except StopIteration:
                break
        
        # Yield y prefetch siguiente
        while self.prefetch_queue:
            batch = self.prefetch_queue.popleft()
            
            # Prefetch siguiente batch
            try:
                next_batch = next(dataloader_iter)
                prefetched = self._prefetch_batch(next_batch)
                self.prefetch_queue.append(prefetched)
            except StopIteration:
                pass
            
            yield batch
    
    def _prefetch_batch(self, batch: dict) -> dict:
        """Prefetch batch a GPU"""
        prefetched = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prefetched[key] = value.to(self.device, non_blocking=True)
            else:
                prefetched[key] = value
        return prefetched


class SmartPrefetch:
    """Prefetch inteligente con predicción"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        lookahead: int = 2
    ):
        """
        Inicializar prefetch inteligente
        
        Args:
            model: Modelo
            device: Dispositivo
            lookahead: Número de batches a predecir
        """
        self.model = model.to(device)
        self.device = device
        self.lookahead = lookahead
        self.prefetch_cache = deque(maxlen=lookahead)
        
        logger.info(f"Smart Prefetch inicializado (lookahead={lookahead})")
    
    def prefetch_and_predict(
        self,
        batches: list
    ) -> list:
        """
        Prefetch y predecir batches
        
        Args:
            batches: Lista de batches
            
        Returns:
            Lista de resultados
        """
        results = []
        
        for i, batch in enumerate(batches):
            # Prefetch batch actual y siguientes
            prefetched_batch = self._prefetch_batch(batch)
            
            # Prefetch siguientes batches si están disponibles
            for j in range(1, self.lookahead + 1):
                if i + j < len(batches):
                    next_batch = batches[i + j]
                    prefetched_next = self._prefetch_batch(next_batch)
                    self.prefetch_cache.append(prefetched_next)
            
            # Inferencia
            with torch.no_grad():
                output = self.model(**prefetched_batch)
                results.append(output)
        
        return results
    
    def _prefetch_batch(self, batch: dict) -> dict:
        """Prefetch batch"""
        prefetched = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prefetched[key] = value.to(self.device, non_blocking=True)
            else:
                prefetched[key] = value
        return prefetched




