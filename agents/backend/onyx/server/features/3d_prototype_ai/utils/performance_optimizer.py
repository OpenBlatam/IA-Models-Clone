"""
Performance Optimizer - Optimizaciones de rendimiento
======================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Optimizador de rendimiento"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.batch_queue: List[Dict[str, Any]] = []
        self.batch_processing = False
    
    def cache_result(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Almacena resultado en caché"""
        self.cache[key] = value
        self.cache_timestamps[key] = datetime.now() + timedelta(seconds=ttl_seconds)
    
    def get_cached(self, key: str) -> Optional[Any]:
        """Obtiene resultado del caché"""
        if key not in self.cache:
            return None
        
        # Verificar expiración
        if datetime.now() > self.cache_timestamps.get(key, datetime.now()):
            del self.cache[key]
            del self.cache_timestamps[key]
            return None
        
        return self.cache[key]
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Limpia el caché"""
        if pattern:
            keys_to_delete = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
        else:
            self.cache.clear()
            self.cache_timestamps.clear()
    
    def record_metric(self, operation: str, duration: float):
        """Registra una métrica de rendimiento"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        
        # Mantener solo últimas 1000 métricas
        if len(self.metrics[operation]) > 1000:
            self.metrics[operation] = self.metrics[operation][-1000:]
    
    def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento"""
        if operation:
            durations = self.metrics.get(operation, [])
            if not durations:
                return {}
            
            return {
                "operation": operation,
                "count": len(durations),
                "avg": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "p95": sorted(durations)[int(len(durations) * 0.95)] if durations else 0
            }
        
        return {
            op: {
                "count": len(durs),
                "avg": sum(durs) / len(durs) if durs else 0,
                "min": min(durs) if durs else 0,
                "max": max(durs) if durs else 0
            }
            for op, durs in self.metrics.items()
        }
    
    def batch_process(self, items: List[Dict[str, Any]], 
                     processor: Callable, batch_size: int = 10):
        """Procesa items en lotes"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = processor(batch)
            results.extend(batch_results)
        
        return results
    
    async def async_batch_process(self, items: List[Dict[str, Any]],
                                 processor: Callable, batch_size: int = 10):
        """Procesa items en lotes de forma asíncrona"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [processor(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    def debounce(self, func: Callable, delay: float = 0.5):
        """Crea una versión debounced de una función"""
        last_call = [None]
        
        @wraps(func)
        def debounced(*args, **kwargs):
            now = time.time()
            if last_call[0] is None or now - last_call[0] >= delay:
                last_call[0] = now
                return func(*args, **kwargs)
        
        return debounced
    
    def throttle(self, func: Callable, limit: int = 10, window: float = 60.0):
        """Crea una versión throttled de una función"""
        calls = []
        
        @wraps(func)
        def throttled(*args, **kwargs):
            now = time.time()
            # Limpiar llamadas fuera de la ventana
            calls[:] = [c for c in calls if now - c < window]
            
            if len(calls) < limit:
                calls.append(now)
                return func(*args, **kwargs)
            else:
                logger.warning(f"Throttle limit reached for {func.__name__}")
                return None
        
        return throttled
    
    def optimize_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimiza parámetros de consulta"""
        optimized = query_params.copy()
        
        # Limitar resultados
        if "limit" not in optimized:
            optimized["limit"] = 50
        elif optimized["limit"] > 100:
            optimized["limit"] = 100
        
        # Agregar índices sugeridos
        if "sort_by" in optimized:
            optimized["use_index"] = optimized["sort_by"]
        
        return optimized


def performance_monitor(func: Callable):
    """Decorador para monitorear rendimiento"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Registrar métrica
            optimizer = PerformanceOptimizer()
            optimizer.record_metric(func.__name__, duration)
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {duration:.3f}s: {e}")
            raise
    
    return wrapper




