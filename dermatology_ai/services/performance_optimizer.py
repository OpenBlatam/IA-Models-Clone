"""
Optimizaciones de rendimiento
"""

import time
import functools
from typing import Callable, Any, Dict
from collections import defaultdict
import threading


class PerformanceMonitor:
    """Monitor de rendimiento"""
    
    def __init__(self):
        """Inicializa el monitor"""
        self.metrics: Dict[str, list] = defaultdict(list)
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float):
        """
        Registra una métrica
        
        Args:
            name: Nombre de la métrica
            value: Valor
        """
        with self.lock:
            self.metrics[name].append({
                "value": value,
                "timestamp": time.time()
            })
            # Mantener solo últimos 1000 registros
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
    
    def get_statistics(self, name: str) -> Dict:
        """
        Obtiene estadísticas de una métrica
        
        Args:
            name: Nombre de la métrica
            
        Returns:
            Diccionario con estadísticas
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = [m["value"] for m in self.metrics[name]]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None
        }
    
    def get_all_statistics(self) -> Dict:
        """Obtiene estadísticas de todas las métricas"""
        return {
            name: self.get_statistics(name)
            for name in self.metrics.keys()
        }


class PerformanceOptimizer:
    """Optimizador de rendimiento"""
    
    def __init__(self):
        """Inicializa el optimizador"""
        self.monitor = PerformanceMonitor()
    
    def time_function(self, func_name: str = None):
        """
        Decorador para medir tiempo de ejecución
        
        Args:
            func_name: Nombre de la función (opcional)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start
                    name = func_name or func.__name__
                    self.monitor.record_metric(f"function.{name}", duration)
            
            return wrapper
        return decorator
    
    def cache_result(self, ttl: int = 3600):
        """
        Decorador para cachear resultados
        
        Args:
            ttl: Tiempo de vida en segundos
        """
        cache = {}
        cache_times = {}
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Crear clave de cache
                cache_key = str(args) + str(sorted(kwargs.items()))
                
                # Verificar cache
                if cache_key in cache:
                    cache_time = cache_times.get(cache_key, 0)
                    if time.time() - cache_time < ttl:
                        return cache[cache_key]
                
                # Ejecutar función
                result = func(*args, **kwargs)
                
                # Guardar en cache
                cache[cache_key] = result
                cache_times[cache_key] = time.time()
                
                return result
            
            return wrapper
        return decorator
    
    def batch_process(self, items: list, batch_size: int = 10,
                     processor: Callable = None) -> list:
        """
        Procesa items en lotes
        
        Args:
            items: Lista de items
            batch_size: Tamaño del lote
            processor: Función procesadora
            
        Returns:
            Lista de resultados
        """
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            if processor:
                batch_results = [processor(item) for item in batch]
            else:
                batch_results = batch
            
            results.extend(batch_results)
        
        return results
    
    def get_performance_report(self) -> Dict:
        """Obtiene reporte de rendimiento"""
        stats = self.monitor.get_all_statistics()
        
        return {
            "timestamp": time.time(),
            "metrics": stats,
            "summary": {
                "total_metrics": len(stats),
                "slowest_operations": self._get_slowest_operations(stats)
            }
        }
    
    def _get_slowest_operations(self, stats: Dict, limit: int = 10) -> list:
        """Obtiene operaciones más lentas"""
        slow_ops = []
        
        for name, stat in stats.items():
            if "avg" in stat:
                slow_ops.append({
                    "name": name,
                    "avg_time": stat["avg"],
                    "max_time": stat.get("max", 0)
                })
        
        slow_ops.sort(key=lambda x: x["avg_time"], reverse=True)
        return slow_ops[:limit]






