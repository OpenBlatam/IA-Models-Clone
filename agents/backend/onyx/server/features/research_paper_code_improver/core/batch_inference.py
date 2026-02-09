"""
Batch Inference Manager - Gestor de inferencia en batch
========================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from collections import deque

from .base_classes import BaseManager, BaseConfig
from .common_utils import get_device, move_to_device, get_model_output
from .constants import DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request de inferencia"""
    request_id: str
    input_data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Mayor número = mayor prioridad


@dataclass
class BatchConfig(BaseConfig):
    """Configuración de batch"""
    max_batch_size: int = DEFAULT_BATCH_SIZE
    timeout_ms: float = 100.0
    max_wait_time_ms: float = 50.0
    device: Optional[str] = None  # Se resuelve con get_device()


class BatchInferenceManager(BaseManager):
    """Gestor de inferencia en batch"""
    
    def __init__(self, model: nn.Module, config: BatchConfig):
        super().__init__(config)
        self.model = model
        self.config = config
        self.device = get_device(config.device)
        self.model = model.to(self.device)
        self.request_queue: deque = deque()
        self.batch_cache: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_latency_ms": 0.0
        }
    
    def add_request(self, request: InferenceRequest):
        """Agrega request a la cola"""
        self.request_queue.append(request)
        self.stats["total_requests"] += 1
    
    def process_batch(self) -> List[Dict[str, Any]]:
        """Procesa un batch de requests"""
        if not self.request_queue:
            return []
        
        # Formar batch
        batch = []
        batch_size = 0
        
        start_time = datetime.now()
        
        while (self.request_queue and 
               batch_size < self.config.max_batch_size and
               (datetime.now() - start_time).total_seconds() * 1000 < self.config.max_wait_time_ms):
            request = self.request_queue.popleft()
            batch.append(request)
            batch_size += 1
        
        if not batch:
            return []
        
        # Procesar batch
        try:
            inputs = [req.input_data for req in batch]
            
            # Convertir a tensor batch
            if isinstance(inputs[0], torch.Tensor):
                batch_tensor = torch.stack(inputs)
            elif isinstance(inputs[0], dict):
                # Para inputs dict, combinar
                batch_dict = {}
                for key in inputs[0].keys():
                    batch_dict[key] = torch.stack([inp[key] for inp in inputs])
                batch_tensor = batch_dict
            else:
                batch_tensor = torch.tensor(inputs)
            
            # Inferencia usando utilidades compartidas
            outputs = get_model_output(self.model, batch_tensor, str(self.device))
            
            # Separar outputs
            if isinstance(outputs, torch.Tensor):
                outputs_list = torch.split(outputs, 1, dim=0)
            elif hasattr(outputs, 'logits'):
                outputs_list = torch.split(outputs.logits, 1, dim=0)
            else:
                outputs_list = [outputs] * len(batch)
            
            # Formar resultados
            results = []
            for i, request in enumerate(batch):
                result = {
                    "request_id": request.request_id,
                    "output": outputs_list[i] if i < len(outputs_list) else None,
                    "latency_ms": (datetime.now() - request.timestamp).total_seconds() * 1000
                }
                results.append(result)
            
            # Actualizar stats
            self.stats["total_batches"] += 1
            self.stats["avg_batch_size"] = (
                (self.stats["avg_batch_size"] * (self.stats["total_batches"] - 1) + batch_size) /
                self.stats["total_batches"]
            )
            
            avg_latency = sum(r["latency_ms"] for r in results) / len(results)
            self.stats["avg_latency_ms"] = (
                (self.stats["avg_latency_ms"] * (self.stats["total_batches"] - 1) + avg_latency) /
                self.stats["total_batches"]
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Error procesando batch: {e}")
            return [{"request_id": req.request_id, "error": str(e)} for req in batch]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        return self.stats.copy()

