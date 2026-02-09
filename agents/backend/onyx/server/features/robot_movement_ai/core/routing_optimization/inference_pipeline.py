"""
Optimized Inference Pipeline
============================

Pipeline de inferencia optimizado con batching, caching y async.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable
from collections import OrderedDict
import asyncio
import logging
from dataclasses import dataclass
import time
import threading
from queue import Queue
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request de inferencia."""
    input_data: torch.Tensor
    request_id: str
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = None


@dataclass
class InferenceResult:
    """Resultado de inferencia."""
    output: torch.Tensor
    request_id: str
    inference_time: float
    metadata: Dict[str, Any] = None


class BatchInferencePipeline:
    """
    Pipeline de inferencia con batching inteligente.
    """
    
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        max_wait_time: float = 0.01,  # 10ms
        device: Optional[str] = None
    ):
        """
        Inicializar pipeline.
        
        Args:
            model: Modelo
            batch_size: Tamaño de batch
            max_wait_time: Tiempo máximo de espera para formar batch (segundos)
            device: Dispositivo
        """
        self.model = model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()
        self.model.to(self.device)
        
        self.request_queue = Queue()
        self.result_cache = OrderedDict()
        self.cache_size = 1000
        self.running = False
        self.worker_thread = None
    
    def _batch_requests(self, requests: List[InferenceRequest]) -> List[List[InferenceRequest]]:
        """
        Agrupar requests en batches.
        
        Args:
            requests: Lista de requests
            
        Returns:
            Lista de batches
        """
        batches = []
        current_batch = []
        
        for request in requests:
            current_batch.append(request)
            
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _process_batch(self, batch: List[InferenceRequest]) -> List[InferenceResult]:
        """
        Procesar batch de requests.
        
        Args:
            batch: Batch de requests
            
        Returns:
            Lista de resultados
        """
        # Stack inputs
        inputs = torch.stack([req.input_data for req in batch]).to(self.device)
        
        # Inferencia
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(inputs)
        inference_time = time.time() - start_time
        
        # Crear resultados
        results = []
        for i, request in enumerate(batch):
            result = InferenceResult(
                output=outputs[i].cpu(),
                request_id=request.request_id,
                inference_time=inference_time / len(batch),
                metadata=request.metadata
            )
            results.append(result)
        
        return results
    
    def _worker(self):
        """Worker thread para procesar requests."""
        pending_requests = []
        last_batch_time = time.time()
        
        while self.running or not self.request_queue.empty() or pending_requests:
            # Recoger nuevos requests
            try:
                while not self.request_queue.empty():
                    request = self.request_queue.get(timeout=0.1)
                    pending_requests.append(request)
            except:
                pass
            
            current_time = time.time()
            time_since_last_batch = current_time - last_batch_time
            
            # Procesar si hay batch completo o timeout
            if (len(pending_requests) >= self.batch_size or 
                (pending_requests and time_since_last_batch >= self.max_wait_time)):
                
                batch = pending_requests[:self.batch_size]
                pending_requests = pending_requests[self.batch_size:]
                
                results = self._process_batch(batch)
                
                # Ejecutar callbacks
                for request, result in zip(batch, results):
                    if request.callback:
                        request.callback(result)
                
                last_batch_time = current_time
    
    def predict(self, input_data: torch.Tensor, 
                request_id: Optional[str] = None,
                callback: Optional[Callable] = None,
                metadata: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """
        Predecir (síncrono).
        
        Args:
            input_data: Datos de entrada
            request_id: ID del request
            callback: Callback (opcional)
            metadata: Metadata (opcional)
            
        Returns:
            Resultado
        """
        if request_id is None:
            request_id = f"req_{time.time()}"
        
        # Verificar cache
        cache_key = self._get_cache_key(input_data)
        if cache_key in self.result_cache:
            cached_result = self.result_cache[cache_key]
            # Mover al final (LRU)
            self.result_cache.move_to_end(cache_key)
            return cached_result
        
        # Procesar directamente para modo síncrono
        request = InferenceRequest(
            input_data=input_data,
            request_id=request_id,
            callback=callback,
            metadata=metadata
        )
        
        result = self._process_batch([request])[0]
        
        # Guardar en cache
        self._add_to_cache(cache_key, result)
        
        return result
    
    async def predict_async(self, input_data: torch.Tensor,
                           request_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """
        Predecir (asíncrono).
        
        Args:
            input_data: Datos de entrada
            request_id: ID del request
            metadata: Metadata (opcional)
            
        Returns:
            Resultado
        """
        if request_id is None:
            request_id = f"req_{time.time()}"
        
        # Verificar cache
        cache_key = self._get_cache_key(input_data)
        if cache_key in self.result_cache:
            cached_result = self.result_cache[cache_key]
            self.result_cache.move_to_end(cache_key)
            return cached_result
        
        # Crear future para resultado
        future = asyncio.Future()
        
        def callback(result: InferenceResult):
            future.set_result(result)
        
        request = InferenceRequest(
            input_data=input_data,
            request_id=request_id,
            callback=callback,
            metadata=metadata
        )
        
        self.request_queue.put(request)
        
        result = await future
        
        # Guardar en cache
        self._add_to_cache(cache_key, result)
        
        return result
    
    def start(self):
        """Iniciar pipeline."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            logger.info("Pipeline de inferencia iniciado")
    
    def stop(self):
        """Detener pipeline."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Pipeline de inferencia detenido")
    
    def _get_cache_key(self, input_data: torch.Tensor) -> str:
        """Generar clave de cache."""
        # Hash del tensor usando hashlib para consistencia
        if isinstance(input_data, torch.Tensor):
            data_bytes = input_data.detach().cpu().numpy().tobytes()
        else:
            data_bytes = str(input_data).encode()
        return hashlib.md5(data_bytes).hexdigest()
    
    def _add_to_cache(self, key: str, result: InferenceResult):
        """Agregar a cache."""
        if len(self.result_cache) >= self.cache_size:
            # Remover LRU
            self.result_cache.popitem(last=False)
        
        self.result_cache[key] = result


class AsyncInferenceServer:
    """
    Servidor de inferencia asíncrono.
    """
    
    def __init__(self, model: nn.Module, batch_size: int = 32):
        """
        Inicializar servidor.
        
        Args:
            model: Modelo
            batch_size: Tamaño de batch
        """
        self.pipeline = BatchInferencePipeline(model, batch_size=batch_size)
        self.pipeline.start()
    
    async def predict(self, input_data: torch.Tensor, **kwargs) -> InferenceResult:
        """
        Predecir (async).
        
        Args:
            input_data: Datos de entrada
            **kwargs: Argumentos adicionales
            
        Returns:
            Resultado
        """
        return await self.pipeline.predict_async(input_data, **kwargs)
    
    def shutdown(self):
        """Cerrar servidor."""
        self.pipeline.stop()

