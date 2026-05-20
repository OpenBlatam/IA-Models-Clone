"""
Document Optimizer - Optimizaciones de Performance
===================================================

Optimizaciones para mejorar el rendimiento del analizador de documentos.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from functools import lru_cache, wraps
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class DocumentCache:
    """Cache optimizado para documentos."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Inicializar cache.
        
        Args:
            max_size: Tamaño máximo del cache
            ttl: Time to live en segundos
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener del cache."""
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        value, expiry = self.cache[key]
        
        if time.time() > expiry:
            del self.cache[key]
            self.miss_count += 1
            return None
        
        self.access_count[key] += 1
        self.hit_count += 1
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Establecer en cache."""
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        ttl = ttl or self.ttl
        self.cache[key] = (value, time.time() + ttl)
    
    def _evict_lru(self):
        """Evict least recently used."""
        if not self.cache:
            return
        
        # Encontrar key con menor acceso
        lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        del self.cache[lru_key]
        del self.access_count[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4)
        }


class BatchOptimizer:
    """Optimizador de procesamiento en batch."""
    
    def __init__(self, optimal_batch_size: int = 32):
        """
        Inicializar optimizador.
        
        Args:
            optimal_batch_size: Tamaño óptimo de batch
        """
        self.optimal_batch_size = optimal_batch_size
        self.performance_history: List[Dict[str, Any]] = []
    
    def calculate_optimal_batch_size(
        self,
        total_items: int,
        avg_processing_time: float,
        target_time_per_batch: float = 1.0
    ) -> int:
        """
        Calcular tamaño óptimo de batch.
        
        Args:
            total_items: Total de items a procesar
            avg_processing_time: Tiempo promedio por item
            target_time_per_batch: Tiempo objetivo por batch
        
        Returns:
            Tamaño óptimo de batch
        """
        if avg_processing_time == 0:
            return self.optimal_batch_size
        
        optimal = int(target_time_per_batch / avg_processing_time)
        optimal = max(1, min(optimal, total_items, 128))  # Limitar entre 1 y 128
        
        return optimal
    
    def should_split_batch(self, batch_size: int, memory_usage: float) -> bool:
        """Determinar si se debe dividir el batch."""
        # Dividir si el batch es muy grande o uso de memoria es alto
        return batch_size > 64 or memory_usage > 0.8
    
    def record_performance(
        self,
        batch_size: int,
        processing_time: float,
        success: bool
    ):
        """Registrar performance de batch."""
        self.performance_history.append({
            "batch_size": batch_size,
            "processing_time": processing_time,
            "success": success,
            "timestamp": time.time()
        })
        
        # Mantener solo últimos 100
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Obtener recomendaciones de optimización."""
        if not self.performance_history:
            return {"recommendations": []}
        
        recent = self.performance_history[-20:]  # Últimos 20
        
        avg_time = np.mean([p["processing_time"] for p in recent])
        avg_batch_size = np.mean([p["batch_size"] for p in recent])
        success_rate = sum(1 for p in recent if p["success"]) / len(recent)
        
        recommendations = []
        
        if avg_time > 5.0:
            recommendations.append("Tiempo de procesamiento alto - considerar reducir batch size")
        
        if success_rate < 0.9:
            recommendations.append("Tasa de éxito baja - revisar errores y timeouts")
        
        if avg_batch_size < 8:
            recommendations.append("Batch size pequeño - considerar aumentar para mejor throughput")
        
        return {
            "avg_processing_time": round(avg_time, 2),
            "avg_batch_size": round(avg_batch_size, 1),
            "success_rate": round(success_rate, 4),
            "recommendations": recommendations
        }


class MemoryOptimizer:
    """Optimizador de uso de memoria."""
    
    def __init__(self, max_memory_mb: int = 2048):
        """
        Inicializar optimizador de memoria.
        
        Args:
            max_memory_mb: Memoria máxima en MB
        """
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
    
    def estimate_memory_usage(
        self,
        document_count: int,
        avg_document_size_kb: float
    ) -> float:
        """
        Estimar uso de memoria.
        
        Args:
            document_count: Número de documentos
            avg_document_size_kb: Tamaño promedio por documento en KB
        
        Returns:
            Uso estimado en MB
        """
        # Estimación: 3x el tamaño del documento (incluye embeddings, tokens, etc.)
        estimated_mb = (document_count * avg_document_size_kb * 3) / 1024
        return estimated_mb
    
    def should_process_in_chunks(
        self,
        document_count: int,
        avg_document_size_kb: float
    ) -> bool:
        """Determinar si procesar en chunks."""
        estimated = self.estimate_memory_usage(document_count, avg_document_size_kb)
        return estimated > self.max_memory_mb * 0.7
    
    def calculate_chunk_size(
        self,
        total_documents: int,
        avg_document_size_kb: float
    ) -> int:
        """Calcular tamaño de chunk óptimo."""
        # Calcular cuántos documentos caben en 70% de la memoria
        available_memory_mb = self.max_memory_mb * 0.7
        chunk_size = int((available_memory_mb * 1024) / (avg_document_size_kb * 3))
        
        return max(1, min(chunk_size, total_documents))


def memoize_async(ttl: int = 3600):
    """Decorator para memoizar funciones async."""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generar key
            key = str(hash(str(args) + str(kwargs)))
            
            # Verificar cache
            if key in cache:
                value, expiry = cache[key]
                if time.time() < expiry:
                    return value
            
            # Ejecutar y cachear
            result = await func(*args, **kwargs)
            cache[key] = (result, time.time() + ttl)
            
            return result
        
        return wrapper
    return decorator


def optimize_model_loading(model_name: str, device: str = "cpu"):
    """Optimizar carga de modelos."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    # Configuraciones de optimización
    config = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
        "device_map": "auto" if device == "cuda" else None
    }
    
    try:
        model = AutoModel.from_pretrained(model_name, **config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        logger.warning(f"Optimización falló, usando método estándar: {e}")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer


__all__ = [
    "DocumentCache",
    "BatchOptimizer",
    "MemoryOptimizer",
    "memoize_async",
    "optimize_model_loading"
]

