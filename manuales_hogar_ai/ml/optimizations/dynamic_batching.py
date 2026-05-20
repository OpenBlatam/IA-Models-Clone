"""
Dynamic Batching
================

Batch size dinámico según carga y recursos.
"""

import logging
import time
from typing import List, Optional, Callable, Any
import asyncio

logger = logging.getLogger(__name__)


class DynamicBatcher:
    """Batcher con tamaño dinámico."""
    
    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        initial_batch_size: int = 8,
        target_latency: float = 0.1,
        adaptation_rate: float = 0.1
    ):
        """
        Inicializar batcher dinámico.
        
        Args:
            min_batch_size: Tamaño mínimo de batch
            max_batch_size: Tamaño máximo de batch
            initial_batch_size: Tamaño inicial
            target_latency: Latencia objetivo (segundos)
            adaptation_rate: Tasa de adaptación
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.target_latency = target_latency
        self.adaptation_rate = adaptation_rate
        
        self.latency_history: List[float] = []
        self._logger = logger
    
    def adapt_batch_size(self, actual_latency: float):
        """
        Adaptar tamaño de batch según latencia.
        
        Args:
            actual_latency: Latencia actual
        """
        self.latency_history.append(actual_latency)
        
        # Mantener solo últimos 10
        if len(self.latency_history) > 10:
            self.latency_history.pop(0)
        
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        
        # Adaptar
        if avg_latency < self.target_latency * 0.8:
            # Latencia baja, aumentar batch
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * (1 + self.adaptation_rate))
            )
        elif avg_latency > self.target_latency * 1.2:
            # Latencia alta, reducir batch
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * (1 - self.adaptation_rate))
            )
        
        self._logger.debug(
            f"Batch size adaptado: {self.current_batch_size} "
            f"(latencia: {avg_latency:.3f}s)"
        )
    
    def get_batch_size(self) -> int:
        """Obtener tamaño de batch actual."""
        return self.current_batch_size
    
    def reset(self):
        """Resetear batcher."""
        self.current_batch_size = self.min_batch_size
        self.latency_history.clear()




