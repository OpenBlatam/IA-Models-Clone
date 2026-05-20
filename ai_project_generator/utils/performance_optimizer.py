"""
Performance Optimizer
====================

Optimizaciones de rendimiento para el generador de proyectos.
"""

import asyncio
import time
import hashlib
import json
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from collections import OrderedDict
import threading
import logging

logger = logging.getLogger(__name__)


class ProjectCache:
    """Caché inteligente para proyectos generados."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.expiry_times: Dict[str, float] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, description: str, project_name: Optional[str] = None, **kwargs) -> str:
        """Genera clave única para proyecto."""
        key_data = {
            'description': description.strip().lower(),
            'project_name': project_name,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Verifica si entrada expiró."""
        if self.ttl is None:
            return False
        if key not in self.expiry_times:
            return False
        return time.time() > self.expiry_times[key]
    
    def get(self, description: str, project_name: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """Obtiene proyecto del cache."""
        key = self._generate_key(description, project_name, **kwargs)
        
        with self.lock:
            if key not in self.cache:
                return None
            
            if self._is_expired(key):
                self._delete(key)
                return None
            
            # Mover al final (LRU)
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            
            return self.cache[key]
    
    def set(self, description: str, project_info: Dict[str, Any], project_name: Optional[str] = None, **kwargs) -> None:
        """Almacena proyecto en cache."""
        key = self._generate_key(description, project_name, **kwargs)
        
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                # Eliminar más antiguo
                oldest_key = next(iter(self.cache))
                self._delete(oldest_key)
            
            self.cache[key] = project_info
            
            if self.ttl is not None:
                self.expiry_times[key] = time.time() + self.ttl
    
    def _delete(self, key: str) -> None:
        """Elimina entrada."""
        self.cache.pop(key, None)
        self.expiry_times.pop(key, None)
        self.access_times.pop(key, None)
    
    def clear(self) -> None:
        """Limpia cache."""
        with self.lock:
            self.cache.clear()
            self.expiry_times.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": self._calculate_hit_rate()
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calcula tasa de aciertos."""
        # Implementación simplificada
        return 0.0


class ParallelProjectProcessor:
    """Procesador paralelo de proyectos."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def process_batch(
        self,
        projects: List[Dict[str, Any]],
        process_func: Callable,
        stop_on_error: bool = False
    ) -> List[Dict[str, Any]]:
        """Procesa múltiples proyectos en paralelo."""
        results = []
        errors = []
        
        async def process_one(project: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with self.semaphore:
                try:
                    result = await process_func(project)
                    return {"project": project, "result": result, "success": True}
                except Exception as e:
                    error_result = {
                        "project": project,
                        "error": str(e),
                        "success": False
                    }
                    if stop_on_error:
                        raise
                    return error_result
        
        tasks = [process_one(project) for project in projects]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separar éxitos y errores
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if isinstance(r, dict) and not r.get("success")]
        
        return {
            "successful": successful,
            "failed": failed,
            "total": len(projects),
            "success_count": len(successful),
            "failed_count": len(failed)
        }


class GenerationOptimizer:
    """Optimizador de generación de proyectos."""
    
    def __init__(self):
        self.generation_stats: Dict[str, Any] = {
            "total_generations": 0,
            "successful": 0,
            "failed": 0,
            "avg_time": 0.0,
            "times": []
        }
        self.lock = threading.RLock()
    
    def record_generation(self, success: bool, duration: float) -> None:
        """Registra estadística de generación."""
        with self.lock:
            self.generation_stats["total_generations"] += 1
            if success:
                self.generation_stats["successful"] += 1
            else:
                self.generation_stats["failed"] += 1
            
            self.generation_stats["times"].append(duration)
            # Mantener solo últimos 100
            if len(self.generation_stats["times"]) > 100:
                self.generation_stats["times"] = self.generation_stats["times"][-100:]
            
            # Calcular promedio
            if self.generation_stats["times"]:
                self.generation_stats["avg_time"] = sum(self.generation_stats["times"]) / len(self.generation_stats["times"])
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas."""
        with self.lock:
            success_rate = (
                self.generation_stats["successful"] / self.generation_stats["total_generations"] * 100
                if self.generation_stats["total_generations"] > 0
                else 0.0
            )
            
            return {
                **self.generation_stats,
                "success_rate": round(success_rate, 2),
                "min_time": min(self.generation_stats["times"]) if self.generation_stats["times"] else 0.0,
                "max_time": max(self.generation_stats["times"]) if self.generation_stats["times"] else 0.0,
            }
    
    def optimize_settings(self) -> Dict[str, Any]:
        """Sugiere optimizaciones basadas en estadísticas."""
        stats = self.get_stats()
        suggestions = []
        
        if stats["avg_time"] > 60:
            suggestions.append("Considerar procesamiento paralelo")
        
        if stats["success_rate"] < 80:
            suggestions.append("Revisar validaciones y manejo de errores")
        
        if len(stats["times"]) > 10:
            variance = sum((t - stats["avg_time"]) ** 2 for t in stats["times"]) / len(stats["times"])
            if variance > 100:
                suggestions.append("Tiempos de generación muy variables, considerar optimización")
        
        return {
            "suggestions": suggestions,
            "stats": stats
        }


class SmartBatchProcessor:
    """Procesador inteligente de lotes."""
    
    def __init__(self, batch_size: int = 10, timeout: float = 5.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.buffer: List[Dict[str, Any]] = []
        self.last_batch_time = time.time()
        self.lock = threading.RLock()
    
    def add(self, item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Agrega item y retorna lote si está completo."""
        with self.lock:
            self.buffer.append(item)
            current_time = time.time()
            
            # Retornar lote si está completo o timeout
            if (len(self.buffer) >= self.batch_size or 
                (current_time - self.last_batch_time) >= self.timeout):
                batch = list(self.buffer)
                self.buffer.clear()
                self.last_batch_time = current_time
                return batch
            return None
    
    def flush(self) -> Optional[List[Dict[str, Any]]]:
        """Fuerza retorno de lote actual."""
        with self.lock:
            if self.buffer:
                batch = list(self.buffer)
                self.buffer.clear()
                return batch
            return None


def cached_generation(cache: Optional[ProjectCache] = None, ttl: Optional[float] = None):
    """Decorador para caché de generación."""
    if cache is None:
        cache = ProjectCache(ttl=ttl)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(description: str, project_name: Optional[str] = None, **kwargs):
            # Intentar obtener del cache
            cached_result = cache.get(description, project_name, **kwargs)
            if cached_result is not None:
                logger.info(f"Cache hit for project: {project_name or 'unnamed'}")
                return cached_result
            
            # Generar proyecto
            result = await func(description, project_name, **kwargs)
            
            # Guardar en cache
            if result:
                cache.set(description, result, project_name, **kwargs)
            
            return result
        
        wrapper.cache = cache
        return wrapper
    return decorator


# Factory functions
_project_cache = None
_generation_optimizer = None

def get_project_cache() -> ProjectCache:
    """Obtiene caché global de proyectos."""
    global _project_cache
    if _project_cache is None:
        _project_cache = ProjectCache(max_size=1000, ttl=3600)
    return _project_cache

def get_generation_optimizer() -> GenerationOptimizer:
    """Obtiene optimizador global."""
    global _generation_optimizer
    if _generation_optimizer is None:
        _generation_optimizer = GenerationOptimizer()
    return _generation_optimizer
