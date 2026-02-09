"""
Optimizador de Rendimiento para Control de Calidad
"""

import logging
import time
from typing import Dict, Optional, Callable, Any
from functools import wraps
from collections import deque
import threading

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Optimizador de rendimiento con caching y optimizaciones
    """
    
    def __init__(self, cache_size: int = 100):
        """
        Inicializar optimizador
        
        Args:
            cache_size: Tamaño del caché
        """
        self.cache_size = cache_size
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.performance_metrics: Dict[str, deque] = {}
        self.lock = threading.Lock()
        
        logger.info(f"Performance Optimizer initialized with cache size {cache_size}")
    
    def cached(self, ttl: float = 60.0):
        """
        Decorador para funciones con caché
        
        Args:
            ttl: Time to live en segundos
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generar clave de caché
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Verificar caché
                with self.lock:
                    if cache_key in self.cache:
                        timestamp = self.cache_timestamps.get(cache_key, 0)
                        if time.time() - timestamp < ttl:
                            logger.debug(f"Cache hit for {func.__name__}")
                            return self.cache[cache_key]
                
                # Ejecutar función
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Guardar en caché
                with self.lock:
                    if len(self.cache) >= self.cache_size:
                        # Eliminar entrada más antigua
                        oldest_key = min(
                            self.cache_timestamps.keys(),
                            key=lambda k: self.cache_timestamps[k]
                        )
                        del self.cache[oldest_key]
                        del self.cache_timestamps[oldest_key]
                    
                    self.cache[cache_key] = result
                    self.cache_timestamps[cache_key] = time.time()
                
                # Registrar métrica
                self._record_metric(func.__name__, execution_time)
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generar clave de caché"""
        import hashlib
        import json
        
        key_data = {
            "func": func_name,
            "args": str(args),
            "kwargs": json.dumps(kwargs, sort_keys=True)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _record_metric(self, func_name: str, execution_time: float):
        """Registrar métrica de rendimiento"""
        if func_name not in self.performance_metrics:
            self.performance_metrics[func_name] = deque(maxlen=100)
        
        self.performance_metrics[func_name].append(execution_time)
    
    def get_performance_stats(self, func_name: Optional[str] = None) -> Dict:
        """
        Obtener estadísticas de rendimiento
        
        Args:
            func_name: Nombre de función (opcional)
            
        Returns:
            Estadísticas de rendimiento
        """
        if func_name:
            if func_name not in self.performance_metrics:
                return {}
            
            times = list(self.performance_metrics[func_name])
            return {
                "function": func_name,
                "call_count": len(times),
                "avg_time": sum(times) / len(times) if times else 0,
                "min_time": min(times) if times else 0,
                "max_time": max(times) if times else 0,
                "total_time": sum(times)
            }
        else:
            # Estadísticas globales
            stats = {
                "cache_size": len(self.cache),
                "cache_hit_rate": 0.0,  # Se calcularía con más información
                "functions": {}
            }
            
            for func_name in self.performance_metrics:
                stats["functions"][func_name] = self.get_performance_stats(func_name)
            
            return stats
    
    def clear_cache(self):
        """Limpiar caché"""
        with self.lock:
            self.cache.clear()
            self.cache_timestamps.clear()
            logger.info("Cache cleared")
    
    def clear_metrics(self):
        """Limpiar métricas"""
        self.performance_metrics.clear()
        logger.info("Performance metrics cleared")


def measure_time(func: Callable) -> Callable:
    """
    Decorador para medir tiempo de ejecución
    
    Args:
        func: Función a medir
        
    Returns:
        Función decorada
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.debug(f"{func.__name__} executed in {execution_time:.4f}s")
        return result
    
    return wrapper






