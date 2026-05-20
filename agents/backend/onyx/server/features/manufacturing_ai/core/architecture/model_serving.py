"""
Model Serving Infrastructure
=============================

Infraestructura avanzada para servir modelos de ML en producción.
"""

import asyncio
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ServingMode(Enum):
    """Modos de serving."""
    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class ServingRequest:
    """Request de serving."""
    request_id: str
    model_id: str
    input_data: Any
    priority: int = 0
    timeout: Optional[float] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ServingResponse:
    """Response de serving."""
    request_id: str
    result: Any
    latency: float
    timestamp: float
    success: bool = True
    error: Optional[str] = None


class ModelServer:
    """Servidor de modelos con múltiples modos."""
    
    def __init__(
        self,
        model: Any,
        model_id: str,
        max_batch_size: int = 32,
        max_queue_size: int = 1000
    ):
        self.model = model
        self.model_id = model_id
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        
        self.request_queue: deque = deque(maxlen=max_queue_size)
        self.response_callbacks: Dict[str, Callable] = {}
        self.lock = threading.RLock()
        self.running = False
        self.worker_thread = None
        
        # Estadísticas
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0.0,
            "current_queue_size": 0
        }
    
    def serve_sync(self, input_data: Any) -> Any:
        """Serving síncrono."""
        start_time = time.time()
        try:
            if TORCH_AVAILABLE:
                self.model.eval()
                with torch.no_grad():
                    if isinstance(input_data, torch.Tensor):
                        result = self.model(input_data)
                    else:
                        result = self.model(torch.tensor(input_data))
            else:
                result = self.model(input_data)
            
            latency = time.time() - start_time
            self._update_stats(True, latency)
            return result
        except Exception as e:
            latency = time.time() - start_time
            self._update_stats(False, latency)
            logger.error(f"Error in sync serving: {e}")
            raise
    
    async def serve_async(self, input_data: Any) -> Any:
        """Serving asíncrono."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.serve_sync, input_data)
    
    def serve_batch(self, batch_data: List[Any]) -> List[Any]:
        """Serving por lotes."""
        start_time = time.time()
        try:
            if TORCH_AVAILABLE:
                self.model.eval()
                with torch.no_grad():
                    if isinstance(batch_data[0], torch.Tensor):
                        batch_tensor = torch.stack(batch_data)
                    else:
                        batch_tensor = torch.tensor(batch_data)
                    results = self.model(batch_tensor)
                    results_list = results.tolist() if hasattr(results, 'tolist') else results
            else:
                results_list = [self.model(item) for item in batch_data]
            
            latency = time.time() - start_time
            self._update_stats(True, latency)
            return results_list
        except Exception as e:
            latency = time.time() - start_time
            self._update_stats(False, latency)
            logger.error(f"Error in batch serving: {e}")
            raise
    
    def queue_request(self, request: ServingRequest, callback: Optional[Callable] = None) -> str:
        """Encola request para procesamiento."""
        with self.lock:
            if len(self.request_queue) >= self.max_queue_size:
                raise RuntimeError("Request queue is full")
            
            self.request_queue.append(request)
            if callback:
                self.response_callbacks[request.request_id] = callback
            
            if not self.running:
                self._start_worker()
        
        return request.request_id
    
    def _start_worker(self) -> None:
        """Inicia worker thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def _worker_loop(self) -> None:
        """Loop del worker."""
        batch = []
        batch_start_time = time.time()
        
        while self.running or self.request_queue:
            try:
                # Agregar requests al batch
                while len(batch) < self.max_batch_size and self.request_queue:
                    request = self.request_queue.popleft()
                    batch.append(request)
                
                # Procesar batch si está lleno o timeout
                if batch and (len(batch) >= self.max_batch_size or 
                             (time.time() - batch_start_time) > 0.1):
                    self._process_batch(batch)
                    batch = []
                    batch_start_time = time.time()
                
                if not batch and not self.request_queue:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                batch = []
    
    def _process_batch(self, batch: List[ServingRequest]) -> None:
        """Procesa lote de requests."""
        try:
            input_data = [req.input_data for req in batch]
            results = self.serve_batch(input_data)
            
            for i, request in enumerate(batch):
                response = ServingResponse(
                    request_id=request.request_id,
                    result=results[i] if i < len(results) else None,
                    latency=time.time() - request.timestamp,
                    timestamp=time.time(),
                    success=True
                )
                
                if request.request_id in self.response_callbacks:
                    callback = self.response_callbacks.pop(request.request_id)
                    callback(response)
        except Exception as e:
            for request in batch:
                response = ServingResponse(
                    request_id=request.request_id,
                    result=None,
                    latency=time.time() - request.timestamp,
                    timestamp=time.time(),
                    success=False,
                    error=str(e)
                )
                
                if request.request_id in self.response_callbacks:
                    callback = self.response_callbacks.pop(request.request_id)
                    callback(response)
    
    def _update_stats(self, success: bool, latency: float) -> None:
        """Actualiza estadísticas."""
        with self.lock:
            self.stats["total_requests"] += 1
            if success:
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1
            
            # Actualizar latencia promedio
            total = self.stats["total_requests"]
            current_avg = self.stats["avg_latency"]
            self.stats["avg_latency"] = (current_avg * (total - 1) + latency) / total
            
            self.stats["current_queue_size"] = len(self.request_queue)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas."""
        with self.lock:
            return self.stats.copy()
    
    def stop(self) -> None:
        """Detiene servidor."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)


class ModelServerRegistry:
    """Registro de servidores de modelos."""
    
    def __init__(self):
        self.servers: Dict[str, ModelServer] = {}
        self.lock = threading.RLock()
    
    def register(self, model_id: str, model: Any, **kwargs) -> ModelServer:
        """Registra modelo."""
        with self.lock:
            if model_id in self.servers:
                raise ValueError(f"Model {model_id} already registered")
            
            server = ModelServer(model, model_id, **kwargs)
            self.servers[model_id] = server
            return server
    
    def get_server(self, model_id: str) -> Optional[ModelServer]:
        """Obtiene servidor."""
        with self.lock:
            return self.servers.get(model_id)
    
    def unregister(self, model_id: str) -> None:
        """Desregistra modelo."""
        with self.lock:
            if model_id in self.servers:
                server = self.servers.pop(model_id)
                server.stop()
    
    def list_models(self) -> List[str]:
        """Lista modelos registrados."""
        with self.lock:
            return list(self.servers.keys())


class LoadBalancer:
    """Balanceador de carga para múltiples servidores."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.servers: List[ModelServer] = []
        self.current_index = 0
        self.lock = threading.RLock()
    
    def add_server(self, server: ModelServer) -> None:
        """Agrega servidor."""
        with self.lock:
            if server not in self.servers:
                self.servers.append(server)
    
    def remove_server(self, server: ModelServer) -> None:
        """Remueve servidor."""
        with self.lock:
            if server in self.servers:
                self.servers.remove(server)
    
    def select_server(self) -> Optional[ModelServer]:
        """Selecciona servidor según estrategia."""
        with self.lock:
            if not self.servers:
                return None
            
            if self.strategy == "round_robin":
                server = self.servers[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.servers)
                return server
            elif self.strategy == "least_loaded":
                return min(self.servers, key=lambda s: s.stats["current_queue_size"])
            else:
                return self.servers[0]
    
    def serve(self, input_data: Any) -> Any:
        """Sirve request usando balanceador."""
        server = self.select_server()
        if server is None:
            raise RuntimeError("No servers available")
        return server.serve_sync(input_data)


# Factory functions
_registry = None

def get_model_server_registry() -> ModelServerRegistry:
    """Obtiene registro global."""
    global _registry
    if _registry is None:
        _registry = ModelServerRegistry()
    return _registry


