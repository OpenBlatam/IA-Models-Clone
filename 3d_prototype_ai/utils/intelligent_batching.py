"""
Intelligent Batching - Batching inteligente para inferencia
=============================================================
Batching adaptativo y dinámico para optimizar throughput
"""

import logging
import torch
from typing import List, Any, Optional, Callable
from collections import deque
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Request para batching"""
    input_data: Any
    callback: Optional[Callable] = None
    priority: int = 0
    timestamp: float = 0.0


class IntelligentBatcher:
    """Batching inteligente para inferencia"""
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
        min_batch_size: int = 1
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        
        self.request_queue: deque = deque()
        self.last_batch_time = time.time()
        self.batch_count = 0
        self.total_processed = 0
    
    def add_request(
        self,
        input_data: Any,
        callback: Optional[Callable] = None,
        priority: int = 0
    ):
        """Agrega request a la cola"""
        request = BatchRequest(
            input_data=input_data,
            callback=callback,
            priority=priority,
            timestamp=time.time()
        )
        
        self.request_queue.append(request)
        
        # Verificar si debemos procesar batch
        if self._should_process_batch():
            return self._process_batch()
        
        return None
    
    def _should_process_batch(self) -> bool:
        """Determina si debemos procesar batch"""
        queue_size = len(self.request_queue)
        time_since_last = time.time() - self.last_batch_time
        
        # Procesar si:
        # 1. Cola llena
        # 2. Tiempo máximo alcanzado y tenemos mínimo
        return (queue_size >= self.max_batch_size or
                (time_since_last >= self.max_wait_time and queue_size >= self.min_batch_size))
    
    def _process_batch(self) -> List[Any]:
        """Procesa batch actual"""
        if not self.request_queue:
            return []
        
        # Tomar requests
        batch_size = min(len(self.request_queue), self.max_batch_size)
        batch_requests = [self.request_queue.popleft() for _ in range(batch_size)]
        
        # Ordenar por prioridad
        batch_requests.sort(key=lambda x: x.priority, reverse=True)
        
        # Extraer inputs
        batch_inputs = [req.input_data for req in batch_requests]
        callbacks = [req.callback for req in batch_requests]
        
        # Retornar batch para procesamiento
        self.batch_count += 1
        self.total_processed += len(batch_inputs)
        self.last_batch_time = time.time()
        
        return {
            "inputs": batch_inputs,
            "callbacks": callbacks,
            "batch_size": len(batch_inputs)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        return {
            "queue_size": len(self.request_queue),
            "batch_count": self.batch_count,
            "total_processed": self.total_processed,
            "avg_batch_size": self.total_processed / self.batch_count if self.batch_count > 0 else 0
        }


class AdaptiveBatcher:
    """Batching adaptativo que ajusta tamaño dinámicamente"""
    
    def __init__(
        self,
        initial_batch_size: int = 8,
        max_batch_size: int = 64,
        min_batch_size: int = 1
    ):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        
        self.latency_history: deque = deque(maxlen=100)
        self.throughput_history: deque = deque(maxlen=100)
    
    def adjust_batch_size(self, latency: float, throughput: float):
        """Ajusta tamaño de batch dinámicamente"""
        self.latency_history.append(latency)
        self.throughput_history.append(throughput)
        
        if len(self.latency_history) < 10:
            return
        
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
        
        # Si latencia es baja y throughput es alto, aumentar batch
        if avg_latency < 0.1 and avg_throughput > 100:
            self.current_batch_size = min(
                self.current_batch_size + 4,
                self.max_batch_size
            )
        # Si latencia es alta, reducir batch
        elif avg_latency > 0.5:
            self.current_batch_size = max(
                self.current_batch_size - 4,
                self.min_batch_size
            )
        
        logger.info(f"Adjusted batch size to: {self.current_batch_size}")




