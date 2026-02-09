"""
Model Serving System - Sistema de serving e inferencia optimizada
==================================================================
Optimización de inferencia, batching, y caching
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    """Configuración de serving"""
    batch_size: int = 32
    max_batch_wait_time: float = 0.1  # segundos
    use_batching: bool = True
    use_caching: bool = True
    cache_size: int = 1000
    use_quantization: bool = False
    use_torchscript: bool = False


class ModelServer:
    """Sistema de serving de modelos"""
    
    def __init__(self, model: nn.Module, config: ServingConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mover modelo a device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Optimizaciones
        if config.use_quantization:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
        
        if config.use_torchscript:
            try:
                self.model = torch.jit.script(self.model)
                logger.info("Model converted to TorchScript")
            except Exception as e:
                logger.warning(f"Failed to convert to TorchScript: {e}")
        
        # Caching
        self.cache: Dict[str, Any] = {}
        self.cache_queue = deque(maxlen=config.cache_size)
        
        # Batching
        self.batch_queue: List[Dict[str, Any]] = []
        self.batch_lock = False
    
    def predict(self, input_data: Any, cache_key: Optional[str] = None) -> Any:
        """Predicción individual"""
        # Verificar cache
        if self.config.use_caching and cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Preparar input
        if isinstance(input_data, torch.Tensor):
            input_tensor = input_data.to(self.device)
        else:
            input_tensor = torch.tensor(input_data).to(self.device)
        
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Inferencia
        with torch.no_grad():
            output = self.model(input_tensor)
        
        result = output.cpu().numpy() if isinstance(output, torch.Tensor) else output
        
        # Guardar en cache
        if self.config.use_caching and cache_key:
            self.cache[cache_key] = result
            self.cache_queue.append(cache_key)
        
        return result
    
    def predict_batch(self, inputs: List[Any]) -> List[Any]:
        """Predicción en batch"""
        # Preparar batch
        batch_tensors = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                batch_tensors.append(inp)
            else:
                batch_tensors.append(torch.tensor(inp))
        
        # Stack batch
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Inferencia
        with torch.no_grad():
            outputs = self.model(batch)
        
        # Convertir a lista
        if isinstance(outputs, torch.Tensor):
            results = outputs.cpu().numpy().tolist()
        else:
            results = outputs
        
        return results
    
    def serve_async(self, input_data: Any, callback: Callable[[Any], None]):
        """Serving asíncrono con batching"""
        if not self.config.use_batching:
            result = self.predict(input_data)
            callback(result)
            return
        
        # Agregar a batch queue
        request = {
            "input": input_data,
            "callback": callback,
            "timestamp": time.time()
        }
        
        self.batch_queue.append(request)
        
        # Procesar batch si está lleno o ha pasado el tiempo
        if len(self.batch_queue) >= self.config.batch_size:
            self._process_batch()
    
    def _process_batch(self):
        """Procesa batch de requests"""
        if not self.batch_queue or self.batch_lock:
            return
        
        self.batch_lock = True
        
        # Tomar batch
        batch = self.batch_queue[:self.config.batch_size]
        self.batch_queue = self.batch_queue[self.config.batch_size:]
        
        # Preparar inputs
        inputs = [req["input"] for req in batch]
        
        # Predecir
        results = self.predict_batch(inputs)
        
        # Llamar callbacks
        for req, result in zip(batch, results):
            req["callback"](result)
        
        self.batch_lock = False
    
    def clear_cache(self):
        """Limpia cache"""
        self.cache.clear()
        self.cache_queue.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de serving"""
        return {
            "cache_size": len(self.cache),
            "batch_queue_size": len(self.batch_queue),
            "device": str(self.device),
            "config": self.config.__dict__
        }




