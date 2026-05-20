"""
Continuous Batching - Batching Continuo
========================================

Sistema de batching continuo para máxima eficiencia.
"""

import logging
import torch
from typing import List, Dict, Any, Optional
from collections import deque
import time
import threading

logger = logging.getLogger(__name__)


class ContinuousBatcher:
    """Batching continuo para inferencia"""
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.01,  # 10ms
        padding_token_id: int = 0
    ):
        """
        Inicializar continuous batcher
        
        Args:
            max_batch_size: Tamaño máximo de batch
            max_wait_time: Tiempo máximo de espera (segundos)
            padding_token_id: ID de token de padding
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.padding_token_id = padding_token_id
        
        self.queue = deque()
        self.lock = threading.Lock()
        self.processing = False
        
        logger.info(f"Continuous Batcher inicializado (max_batch={max_batch_size})")
    
    def add_request(
        self,
        input_ids: torch.Tensor,
        callback: callable
    ):
        """
        Agregar request al batch
        
        Args:
            input_ids: IDs de input
            callback: Callback para resultado
        """
        with self.lock:
            self.queue.append({
                "input_ids": input_ids,
                "callback": callback,
                "timestamp": time.time()
            })
    
    def get_batch(self) -> Optional[Dict[str, Any]]:
        """
        Obtener batch listo para procesar
        
        Returns:
            Batch o None si no hay suficientes requests
        """
        with self.lock:
            if not self.queue:
                return None
            
            # Obtener requests antiguos
            current_time = time.time()
            batch_requests = []
            
            for request in list(self.queue):
                age = current_time - request["timestamp"]
                
                # Agregar si es viejo o si el batch está lleno
                if age >= self.max_wait_time or len(batch_requests) >= self.max_batch_size:
                    batch_requests.append(request)
                    self.queue.remove(request)
                    
                    if len(batch_requests) >= self.max_batch_size:
                        break
            
            if not batch_requests:
                return None
            
            # Crear batch
            max_len = max(r["input_ids"].size(0) for r in batch_requests)
            batch_inputs = []
            
            for request in batch_requests:
                input_ids = request["input_ids"]
                pad_size = max_len - input_ids.size(0)
                
                if pad_size > 0:
                    padding = torch.full((pad_size,), self.padding_token_id, dtype=input_ids.dtype)
                    input_ids = torch.cat([input_ids, padding])
                
                batch_inputs.append(input_ids)
            
            batched_input = torch.stack(batch_inputs)
            
            return {
                "input_ids": batched_input,
                "callbacks": [r["callback"] for r in batch_requests],
                "attention_mask": (batched_input != self.padding_token_id).long()
            }
    
    def process_batch(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Procesar batch
        
        Args:
            model: Modelo
            batch: Batch de datos
            device: Dispositivo
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Separar resultados y llamar callbacks
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        
        for i, callback in enumerate(batch["callbacks"]):
            result = logits[i] if logits.dim() > 1 else logits
            callback(result.cpu())


class AsyncContinuousBatcher(ContinuousBatcher):
    """Continuous batcher asíncrono"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread = None
    
    def start(self, model: torch.nn.Module, device: str = "cuda"):
        """Iniciar procesamiento asíncrono"""
        self.processing = True
        
        def process_loop():
            while self.processing:
                batch = self.get_batch()
                if batch:
                    self.process_batch(model, batch, device)
                else:
                    time.sleep(0.001)  # Pequeña pausa
        
        self.thread = threading.Thread(target=process_loop, daemon=True)
        self.thread.start()
        logger.info("Async Continuous Batcher iniciado")
    
    def stop(self):
        """Detener procesamiento"""
        self.processing = False
        if self.thread:
            self.thread.join()




