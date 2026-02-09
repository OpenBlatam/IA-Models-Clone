"""
Batch Optimization
==================

Optimizaciones avanzadas para procesamiento por batches.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


class DynamicBatching:
    """
    Batching dinámico para máxima throughput.
    """
    
    def __init__(
        self,
        model: nn.Module,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        target_latency: float = 0.01,  # 10ms
        device: str = "cuda"
    ):
        """
        Inicializar batching dinámico.
        
        Args:
            model: Modelo
            min_batch_size: Batch mínimo
            max_batch_size: Batch máximo
            target_latency: Latencia objetivo (segundos)
            device: Dispositivo
        """
        self.model = model
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency = target_latency
        self.device = device
        
        self.current_batch_size = min_batch_size
        self.latency_history = deque(maxlen=100)
    
    def _measure_latency(self, batch_size: int) -> float:
        """
        Medir latencia para un batch size.
        
        Args:
            batch_size: Tamaño de batch
            
        Returns:
            Latencia en segundos
        """
        dummy_input = torch.randn(batch_size, 20).to(self.device)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        return time.time() - start
    
    def optimize_batch_size(self) -> int:
        """
        Optimizar batch size dinámicamente.
        
        Returns:
            Batch size óptimo
        """
        # Medir latencia actual
        current_latency = self._measure_latency(self.current_batch_size)
        self.latency_history.append(current_latency)
        
        # Ajustar batch size
        if current_latency < self.target_latency:
            # Puede aumentar batch size
            if self.current_batch_size < self.max_batch_size:
                new_batch_size = min(
                    self.current_batch_size * 2,
                    self.max_batch_size
                )
                new_latency = self._measure_latency(new_batch_size)
                
                if new_latency <= self.target_latency:
                    self.current_batch_size = new_batch_size
                    logger.info(f"Batch size aumentado a {new_batch_size}")
        else:
            # Debe reducir batch size
            if self.current_batch_size > self.min_batch_size:
                self.current_batch_size = max(
                    self.current_batch_size // 2,
                    self.min_batch_size
                )
                logger.info(f"Batch size reducido a {self.current_batch_size}")
        
        return self.current_batch_size
    
    def get_optimal_batch_size(self) -> int:
        """
        Obtener batch size óptimo actual.
        
        Returns:
            Batch size
        """
        return self.current_batch_size


class PinnedMemoryBatchProcessor:
    """
    Procesador de batches con pinned memory.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Inicializar procesador.
        
        Args:
            model: Modelo
            device: Dispositivo
        """
        self.model = model
        self.device = device
        
        if device == "cuda":
            # Pre-allocar pinned memory
            self.pinned_buffer = torch.empty(
                (128, 20),
                dtype=torch.float32,
                pin_memory=True
            )
    
    def process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Procesar batch con pinned memory.
        
        Args:
            batch: Batch de inputs
            
        Returns:
            Batch de outputs
        """
        if self.device == "cuda" and batch.device.type == "cpu":
            # Transferir usando pinned memory
            batch = batch.pin_memory()
            batch = batch.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            output = self.model(batch)
        
        return output.cpu()


class BatchQueue:
    """
    Cola de batches optimizada.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Inicializar cola.
        
        Args:
            max_size: Tamaño máximo
        """
        self.queue = deque(maxlen=max_size)
        self.total_batches = 0
    
    def add_batch(self, batch: torch.Tensor):
        """Agregar batch."""
        self.queue.append(batch)
        self.total_batches += 1
    
    def get_batches(self, num_batches: int) -> List[torch.Tensor]:
        """
        Obtener batches.
        
        Args:
            num_batches: Número de batches
            
        Returns:
            Lista de batches
        """
        batches = []
        for _ in range(min(num_batches, len(self.queue))):
            if self.queue:
                batches.append(self.queue.popleft())
        return batches
    
    def merge_batches(self, batches: List[torch.Tensor]) -> torch.Tensor:
        """
        Fusionar batches.
        
        Args:
            batches: Lista de batches
            
        Returns:
            Batch fusionado
        """
        if not batches:
            return torch.empty(0)
        
        return torch.cat(batches, dim=0)

