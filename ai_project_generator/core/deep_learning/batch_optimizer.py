"""
Batch Optimizer - Optimizaciones de batch processing
=====================================================

Utilidades para optimizar procesamiento por batches:
- Dynamic batching
- Batch padding optimizado
- Memory-efficient batching
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from collections import deque

logger = logging.getLogger(__name__)


class BatchOptimizer:
    """Optimizador de batches para inferencia rápida"""
    
    def __init__(self):
        """Inicializa el optimizador de batches"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de optimización de batches.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        perf_dir = utils_dir / "performance"
        perf_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_dynamic_batching(perf_dir, keywords, project_info)
        self._generate_batch_utils(perf_dir, keywords, project_info)
    
    def _generate_dynamic_batching(
        self,
        perf_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de dynamic batching"""
        
        batching_content = '''"""
Dynamic Batching - Batching dinámico para inferencia rápida
============================================================

Sistema de batching dinámico que agrupa requests para máxima throughput.
"""

import torch
from typing import List, Any, Optional, Callable, Dict
from collections import deque
import time
import threading
import logging

logger = logging.getLogger(__name__)


class DynamicBatcher:
    """
    Batching dinámico para agrupar múltiples requests.
    
    Optimiza throughput agrupando requests que llegan cerca en el tiempo.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        max_batch_size: int = 64,  # Aumentado para máxima velocidad
        max_wait_time: float = 0.005,  # 5ms - más agresivo
        device: str = "cuda",
    ):
        """
        Inicializa el dynamic batcher.
        
        Args:
            model: Modelo a usar
            max_batch_size: Tamaño máximo de batch
            max_wait_time: Tiempo máximo de espera (segundos)
            device: Dispositivo a usar
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.device = device
        
        self.queue: deque = deque()
        self.lock = threading.Lock()
        self.processing = False
        
        # Estadísticas
        self.stats = {
            "total_batches": 0,
            "total_requests": 0,
            "avg_batch_size": 0.0,
        }
    
    def _pad_batch(
        self,
        items: List[torch.Tensor],
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Padea batch para que todos tengan el mismo tamaño.
        
        Args:
            items: Lista de tensores
            pad_value: Valor de padding
        
        Returns:
            Tensor con batch padeado
        """
        if not items:
            return torch.empty(0)
        
        # Encontrar tamaño máximo
        max_size = max(item.shape[0] if item.dim() > 0 else 1 for item in items)
        
        # Padear cada item
        padded = []
        for item in items:
            if item.dim() == 0:
                item = item.unsqueeze(0)
            
            current_size = item.shape[0]
            if current_size < max_size:
                padding = torch.full(
                    (max_size - current_size, *item.shape[1:]),
                    pad_value,
                    dtype=item.dtype,
                    device=item.device,
                )
                item = torch.cat([item, padding], dim=0)
            
            padded.append(item)
        
        return torch.stack(padded)
    
    def _process_batch(
        self,
        batch: List[Any],
        callbacks: List[Callable],
    ) -> None:
        """
        Procesa un batch de items.
        
        Args:
            batch: Batch de items
            callbacks: Callbacks para cada item
        """
        try:
            # Preparar batch
            if isinstance(batch[0], torch.Tensor):
                # Padear si es necesario
                batch_tensor = self._pad_batch(batch)
                batch_tensor = batch_tensor.to(self.device)
            elif isinstance(batch[0], dict):
                # Batch de diccionarios (común en transformers)
                batch_tensor = {}
                for key in batch[0].keys():
                    values = [item[key] for item in batch]
                    batch_tensor[key] = self._pad_batch(values).to(self.device)
            else:
                batch_tensor = batch
            
            # Inferencia ultra optimizada
            self.model.eval()
            with torch.no_grad():
                # Usar autocast más agresivo
                with torch.autocast(
                    device_type=self.device,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    enabled=self.device == "cuda",
                    cache_enabled=True,  # Habilitar caché de autocast
                ):
                    # Sincronización solo al final para máxima velocidad
                    outputs = self.model(batch_tensor)
                    
                    # Sincronizar solo si es necesario
                    if self.device == "cuda" and len(batch) >= self.max_batch_size:
                        torch.cuda.synchronize()
            
            # Procesar outputs y llamar callbacks
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.cpu()
                for i, callback in enumerate(callbacks):
                    if i < outputs.shape[0]:
                        callback(outputs[i])
            elif isinstance(outputs, (list, tuple)):
                for i, callback in enumerate(callbacks):
                    if i < len(outputs):
                        callback(outputs[i])
            else:
                for callback in callbacks:
                    callback(outputs)
        
        except Exception as e:
            logger.error(f"Error procesando batch: {e}")
            for callback in callbacks:
                callback(None)
    
    def add_request(
        self,
        item: Any,
        callback: Callable,
    ) -> None:
        """
        Agrega un request al batcher.
        
        Args:
            item: Item a procesar
            callback: Callback con el resultado
        """
        with self.lock:
            self.queue.append((item, callback, time.time()))
            self.stats["total_requests"] += 1
            
            # Procesar si el batch está lleno
            if len(self.queue) >= self.max_batch_size:
                self._process_pending()
            elif not self.processing:
                # Iniciar timer para procesar después de max_wait_time
                threading.Timer(
                    self.max_wait_time,
                    self._process_pending
                ).start()
    
    def _process_pending(self) -> None:
        """Procesa requests pendientes"""
        with self.lock:
            if self.processing or len(self.queue) == 0:
                return
            
            self.processing = True
            
            # Obtener batch
            batch = []
            callbacks = []
            current_time = time.time()
            
            while len(batch) < self.max_batch_size and len(self.queue) > 0:
                item, callback, request_time = self.queue.popleft()
                
                # Agregar si no ha pasado mucho tiempo
                if current_time - request_time <= self.max_wait_time * 2:
                    batch.append(item)
                    callbacks.append(callback)
                else:
                    # Procesar item individual si esperó mucho
                    self._process_batch([item], [callback])
            
            if batch:
                self._process_batch(batch, callbacks)
                self.stats["total_batches"] += 1
                self.stats["avg_batch_size"] = (
                    self.stats["total_requests"] / self.stats["total_batches"]
                    if self.stats["total_batches"] > 0 else 0
                )
            
            self.processing = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del batcher.
        
        Returns:
            Diccionario con estadísticas
        """
        return self.stats.copy()
'''
        
        (perf_dir / "dynamic_batching.py").write_text(batching_content, encoding="utf-8")
    
    def _generate_batch_utils(
        self,
        perf_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de batch"""
        
        batch_utils_content = '''"""
Batch Utilities - Utilidades para batch processing
===================================================

Funciones helper para optimizar procesamiento por batches.
"""

import torch
from typing import List, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def collate_batch(
    items: List[Any],
    pad_value: float = 0.0,
    pad_to_max: bool = True,
) -> torch.Tensor:
    """
    Colatea items en un batch optimizado.
    
    Args:
        items: Lista de items
        pad_value: Valor de padding
        pad_to_max: Si padear al máximo
    
    Returns:
        Batch tensor
    """
    if not items:
        return torch.empty(0)
    
    if isinstance(items[0], torch.Tensor):
        if pad_to_max:
            # Encontrar tamaño máximo
            max_size = max(item.shape[0] if item.dim() > 0 else 1 for item in items)
            
            # Padear
            padded = []
            for item in items:
                if item.dim() == 0:
                    item = item.unsqueeze(0)
                
                current_size = item.shape[0]
                if current_size < max_size:
                    padding = torch.full(
                        (max_size - current_size, *item.shape[1:]),
                        pad_value,
                        dtype=item.dtype,
                        device=item.device,
                    )
                    item = torch.cat([item, padding], dim=0)
                
                padded.append(item)
            
            return torch.stack(padded)
        else:
            return torch.stack(items)
    
    return torch.tensor(items)


def process_batch_fast(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: str = "cuda",
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Procesa batch de forma optimizada.
    
    Args:
        model: Modelo a usar
        batch: Batch de datos
        device: Dispositivo a usar
        use_amp: Si usar mixed precision
    
    Returns:
        Outputs del modelo
    """
    model.eval()
    batch = batch.to(device)
    
    with torch.no_grad():
        if use_amp and device == "cuda":
            with torch.autocast(device_type=device, dtype=torch.float16):
                if device == "cuda":
                    torch.cuda.synchronize()
                
                output = model(batch)
                
                if device == "cuda":
                    torch.cuda.synchronize()
        else:
            if device == "cuda":
                torch.cuda.synchronize()
            
            output = model(batch)
            
            if device == "cuda":
                torch.cuda.synchronize()
    
    return output


def split_large_batch(
    batch: torch.Tensor,
    max_size: int = 32,
) -> List[torch.Tensor]:
    """
    Divide batch grande en batches más pequeños.
    
    Args:
        batch: Batch grande
        max_size: Tamaño máximo por batch
    
    Returns:
        Lista de batches
    """
    batches = []
    batch_size = batch.shape[0]
    
    for i in range(0, batch_size, max_size):
        end = min(i + max_size, batch_size)
        batches.append(batch[i:end])
    
    return batches
'''
        
        (perf_dir / "batch_utils.py").write_text(batch_utils_content, encoding="utf-8")

