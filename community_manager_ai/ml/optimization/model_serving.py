"""
Model Serving - Servicio de Modelos
====================================

Sistema de serving optimizado para producción.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from functools import lru_cache
import threading
from queue import Queue
import time

logger = logging.getLogger(__name__)


class ModelServer:
    """Servidor de modelos optimizado"""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1
    ):
        """
        Inicializar servidor de modelos
        
        Args:
            model: Modelo a servir
            device: Dispositivo
            max_batch_size: Tamaño máximo de batch
            max_wait_time: Tiempo máximo de espera para batching
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        self.model = model.to(self.device)
        self.model.eval()
        
        # Optimizar modelo
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass
        
        # Queue para batching dinámico
        self.request_queue = Queue()
        self.result_queue = Queue()
        self.processing = False
        
        logger.info(f"Model Server inicializado en {self.device}")
    
    def serve_async(self):
        """Iniciar procesamiento asíncrono"""
        self.processing = True
        thread = threading.Thread(target=self._process_queue, daemon=True)
        thread.start()
    
    def _process_queue(self):
        """Procesar queue de requests"""
        batch = []
        last_batch_time = time.time()
        
        while self.processing:
            try:
                # Obtener request con timeout
                try:
                    request = self.request_queue.get(timeout=self.max_wait_time)
                    batch.append(request)
                except:
                    pass
                
                # Procesar batch si está lleno o ha pasado el tiempo
                current_time = time.time()
                should_process = (
                    len(batch) >= self.max_batch_size or
                    (len(batch) > 0 and current_time - last_batch_time >= self.max_wait_time)
                )
                
                if should_process and batch:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"Error procesando queue: {e}")
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Procesar batch de requests"""
        try:
            # Combinar inputs
            combined_inputs = self._combine_batch([r["input"] for r in batch])
            combined_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in combined_inputs.items()
            }
            
            # Inferencia
            with torch.cuda.amp.autocast() if self.device == "cuda" else torch.no_grad():
                outputs = self.model(**combined_inputs)
            
            # Separar resultados
            if hasattr(outputs, "logits"):
                predictions = outputs.logits
            else:
                predictions = outputs
            
            # Enviar resultados
            for i, request in enumerate(batch):
                result = predictions[i] if predictions.dim() > 1 else predictions
                request["callback"](result)
                
        except Exception as e:
            logger.error(f"Error procesando batch: {e}")
            for request in batch:
                request["callback"](None)
    
    def _combine_batch(self, inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combinar batch de inputs"""
        combined = {}
        for key in inputs[0].keys():
            tensors = [inp[key] for inp in inputs]
            combined[key] = torch.stack(tensors)
        return combined
    
    def predict(self, input_data: Dict[str, torch.Tensor], timeout: float = 5.0) -> Optional[torch.Tensor]:
        """
        Predecir (síncrono)
        
        Args:
            input_data: Datos de input
            timeout: Timeout en segundos
            
        Returns:
            Predicción o None
        """
        result_container = {"result": None, "done": False}
        
        def callback(result):
            result_container["result"] = result
            result_container["done"] = True
        
        # Agregar a queue
        self.request_queue.put({
            "input": input_data,
            "callback": callback
        })
        
        # Esperar resultado
        start_time = time.time()
        while not result_container["done"]:
            if time.time() - start_time > timeout:
                logger.warning("Timeout en predicción")
                return None
            time.sleep(0.01)
        
        return result_container["result"]


class ModelPool:
    """Pool de modelos para carga balanceada"""
    
    def __init__(
        self,
        model_factory: callable,
        pool_size: int = 4,
        device: Optional[str] = None
    ):
        """
        Inicializar pool de modelos
        
        Args:
            model_factory: Función que crea modelos
            pool_size: Tamaño del pool
            device: Dispositivo
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [model_factory() for _ in range(pool_size)]
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        self.current_index = 0
        self.lock = threading.Lock()
        logger.info(f"Model Pool inicializado con {pool_size} modelos")
    
    def get_model(self) -> nn.Module:
        """Obtener modelo del pool (round-robin)"""
        with self.lock:
            model = self.models[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.models)
            return model




