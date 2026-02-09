"""
Model serving avanzado con batching optimizado y load balancing
"""

import torch
import torch.nn as nn
import asyncio
from typing import Dict, Any, List, Optional
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Procesador de batches optimizado"""
    
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.device = device
        self.queue = asyncio.Queue()
        self.processing = False
    
    async def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Procesa batch de requests"""
        # Preparar batch
        batched_inputs = self._prepare_batch(batch)
        
        # Inferencia
        with torch.inference_mode():
            outputs = self.model(**batched_inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probabilities = torch.softmax(logits, dim=-1)
        
        # Separar resultados
        results = []
        for i in range(len(batch)):
            results.append({
                "logits": logits[i].cpu().tolist(),
                "probabilities": probabilities[i].cpu().tolist()
            })
        
        return results
    
    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepara batch de inputs"""
        batched = {}
        
        for key in batch[0].keys():
            tensors = [item[key] for item in batch if isinstance(item[key], torch.Tensor)]
            if tensors:
                batched[key] = torch.stack(tensors).to(self.device)
        
        return batched
    
    async def process_queue(self):
        """Procesa queue de requests"""
        while self.processing:
            batch = []
            start_time = time.time()
            
            # Acumular requests
            while len(batch) < self.max_batch_size:
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.max_wait_time
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    if batch:
                        break
                    continue
            
            if batch:
                # Procesar batch
                results = await self.process_batch(batch)
                
                # Enviar resultados
                for item, result in zip(batch, results):
                    if 'future' in item:
                        item['future'].set_result(result)


class LoadBalancedModelServer:
    """Servidor con load balancing"""
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = "cuda"
    ):
        self.models = [m.to(device) for m in models]
        self.device = device
        self.current_model_idx = 0
        self.request_counts = [0] * len(models)
    
    def get_model(self) -> nn.Module:
        """Obtiene modelo usando round-robin"""
        model = self.models[self.current_model_idx]
        self.request_counts[self.current_model_idx] += 1
        self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
        return model
    
    def get_least_loaded_model(self) -> nn.Module:
        """Obtiene modelo menos cargado"""
        least_loaded_idx = min(range(len(self.request_counts)), key=lambda i: self.request_counts[i])
        self.request_counts[least_loaded_idx] += 1
        return self.models[least_loaded_idx]
    
    async def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Predicción con load balancing"""
        model = self.get_least_loaded_model()
        
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probabilities = torch.softmax(logits, dim=-1)
            
            return {
                "logits": logits.cpu().tolist(),
                "probabilities": probabilities.cpu().tolist()
            }




