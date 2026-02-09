"""
Optimizaciones Avanzadas de Rendimiento
========================================
Optimizaciones y mejoras de rendimiento
"""

from typing import Dict, Any, List, Optional, Callable
from functools import lru_cache
import structlog
import asyncio
from collections import OrderedDict, defaultdict
import time

logger = structlog.get_logger()


class LRUCache:
    """Cache LRU simple"""
    
    def __init__(self, max_size: int = 128):
        """
        Inicializar cache
        
        Args:
            max_size: Tamaño máximo del cache
        """
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        logger.info("LRUCache initialized", max_size=max_size)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del cache
        
        Args:
            key: Clave
            
        Returns:
            Valor o None
        """
        if key in self._cache:
            # Mover al final (más reciente)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Guardar valor en cache
        
        Args:
            key: Clave
            value: Valor
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                # Eliminar el más antiguo
                self._cache.popitem(last=False)
        
        self._cache[key] = value
    
    def clear(self) -> None:
        """Limpiar cache"""
        self._cache.clear()
    
    def size(self) -> int:
        """Obtener tamaño del cache"""
        return len(self._cache)


class PerformanceMonitor:
    """Monitor de rendimiento"""
    
    def __init__(self):
        """Inicializar monitor"""
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}
        logger.info("PerformanceMonitor initialized")
    
    def start_timer(self, operation: str) -> None:
        """
        Iniciar timer para operación
        
        Args:
            operation: Nombre de la operación
        """
        self._start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """
        Finalizar timer y registrar métrica
        
        Args:
            operation: Nombre de la operación
            
        Returns:
            Duración en segundos
        """
        if operation not in self._start_times:
            return 0.0
        
        duration = time.time() - self._start_times[operation]
        self._metrics[operation].append(duration)
        del self._start_times[operation]
        
        return duration
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas
        
        Args:
            operation: Operación específica (opcional)
            
        Returns:
            Estadísticas
        """
        if operation:
            if operation not in self._metrics:
                return {}
            
            durations = self._metrics[operation]
            return {
                "operation": operation,
                "count": len(durations),
                "total": sum(durations),
                "average": sum(durations) / len(durations) if durations else 0.0,
                "min": min(durations) if durations else 0.0,
                "max": max(durations) if durations else 0.0
            }
        
        stats = {}
        for op, durations in self._metrics.items():
            stats[op] = {
                "count": len(durations),
                "average": sum(durations) / len(durations) if durations else 0.0,
                "min": min(durations) if durations else 0.0,
                "max": max(durations) if durations else 0.0
            }
        
        return stats


class AsyncBatchProcessor:
    """Procesador asíncrono por lotes optimizado"""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        """
        Inicializar procesador
        
        Args:
            batch_size: Tamaño del lote
            max_concurrent: Máximo concurrente
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        logger.info(
            "AsyncBatchProcessor initialized",
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )
    
    async def process_batch(
        self,
        items: List[Any],
        processor: Callable[[Any], Any]
    ) -> List[Any]:
        """
        Procesar lote de items
        
        Args:
            items: Lista de items
            processor: Función procesadora
            
        Returns:
            Lista de resultados
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_item(item: Any) -> Any:
            async with semaphore:
                if asyncio.iscoroutinefunction(processor):
                    return await processor(item)
                else:
                    return processor(item)
        
        # Procesar en lotes
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await asyncio.gather(
                *[process_item(item) for item in batch],
                return_exceptions=True
            )
            results.extend(batch_results)
        
        return results


# Instancias globales
lru_cache = LRUCache(max_size=256)
performance_monitor = PerformanceMonitor()
async_batch_processor = AsyncBatchProcessor()

