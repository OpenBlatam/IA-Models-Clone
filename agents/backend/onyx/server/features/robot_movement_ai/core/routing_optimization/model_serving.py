"""
Model Serving
=============

Sistema de serving optimizado para producción.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass
import time
from collections import deque, OrderedDict
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI no disponible")


@dataclass
class ServingConfig:
    """Configuración de serving."""
    batch_size: int = 32
    max_queue_size: int = 1000
    timeout: float = 1.0
    num_workers: int = 1
    use_batching: bool = True
    use_cache: bool = True
    cache_size: int = 1000


class ModelServer:
    """
    Servidor de modelo optimizado.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[ServingConfig] = None
    ):
        """
        Inicializar servidor.
        
        Args:
            model: Modelo a servir
            config: Configuración (opcional)
        """
        self.model = model
        self.config = config or ServingConfig()
        
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Cache
        self.cache = OrderedDict() if self.config.use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Estadísticas
        self.request_count = 0
        self.total_inference_time = 0.0
        self.request_times = deque(maxlen=1000)
        
        # Queue para batching
        self.request_queue = Queue(maxsize=self.config.max_queue_size)
        self.workers = []
        self.running = False
    
    def _get_cache_key(self, input_data: Any) -> str:
        """Generar clave de cache."""
        import hashlib
        if isinstance(input_data, torch.Tensor):
            data_bytes = input_data.detach().cpu().numpy().tobytes()
        else:
            data_bytes = str(input_data).encode()
        return hashlib.md5(data_bytes).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[torch.Tensor]:
        """Obtener de cache."""
        if not self.config.use_cache or self.cache is None:
            return None
        
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache_hits += 1
            return self.cache[key]
        
        self.cache_misses += 1
        return None
    
    def _add_to_cache(self, key: str, value: torch.Tensor):
        """Agregar a cache."""
        if not self.config.use_cache or self.cache is None:
            return
        
        if len(self.cache) >= self.config.cache_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Predecir (síncrono).
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            Predicción
        """
        start_time = time.time()
        
        # Verificar cache
        cache_key = self._get_cache_key(input_data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Preparar input
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)
        
        input_data = input_data.to(next(self.model.parameters()).device)
        
        # Inferencia
        with torch.no_grad():
            output = self.model(input_data)
        
        # Guardar en cache
        self._add_to_cache(cache_key, output.cpu())
        
        # Estadísticas
        inference_time = time.time() - start_time
        self.request_count += 1
        self.total_inference_time += inference_time
        self.request_times.append(inference_time)
        
        return output.cpu()
    
    def predict_batch(self, input_batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Predecir batch.
        
        Args:
            input_batch: Batch de inputs
            
        Returns:
            Batch de predicciones
        """
        # Stack inputs
        stacked = torch.stack(input_batch).to(next(self.model.parameters()).device)
        
        # Inferencia
        with torch.no_grad():
            outputs = self.model(stacked)
        
        return [outputs[i].cpu() for i in range(len(input_batch))]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas.
        
        Returns:
            Diccionario con estadísticas
        """
        avg_time = (self.total_inference_time / self.request_count 
                   if self.request_count > 0 else 0.0)
        
        cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses)
                         if (self.cache_hits + self.cache_misses) > 0 else 0.0)
        
        return {
            "total_requests": self.request_count,
            "avg_inference_time_ms": avg_time * 1000,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "throughput_req_per_sec": 1.0 / avg_time if avg_time > 0 else 0.0
        }


if FASTAPI_AVAILABLE:
    def create_fastapi_server(model: nn.Module, 
                              config: Optional[ServingConfig] = None) -> FastAPI:
        """
        Crear servidor FastAPI.
        
        Args:
            model: Modelo
            config: Configuración
            
        Returns:
            App FastAPI
        """
        app = FastAPI(title="Routing AI Model Server")
        server = ModelServer(model, config)
        
        @app.post("/predict")
        async def predict(request: Request):
            """Endpoint de predicción."""
            try:
                data = await request.json()
                input_data = torch.tensor(data["input"], dtype=torch.float32)
                
                output = server.predict(input_data)
                
                return JSONResponse({
                    "success": True,
                    "output": output.tolist(),
                    "stats": server.get_stats()
                })
            except Exception as e:
                return JSONResponse(
                    {"success": False, "error": str(e)},
                    status_code=500
                )
        
        @app.get("/stats")
        async def stats():
            """Endpoint de estadísticas."""
            return JSONResponse(server.get_stats())
        
        @app.get("/health")
        async def health():
            """Health check."""
            return {"status": "healthy"}
        
        return app

