"""
Memory Optimizations - Optimizaciones de memoria
===============================================

Optimizaciones para reducir uso de memoria y mejorar eficiencia.
"""

import logging
import sys
import gc
from typing import Optional, Callable, Any
from functools import lru_cache
import weakref

logger = logging.getLogger(__name__)


class LazyLoader:
    """
    Lazy loader para módulos pesados.
    
    Carga módulos solo cuando se necesitan.
    """
    
    def __init__(self):
        self._modules = {}
        self._loaders = {}
    
    def register(self, name: str, loader: Callable):
        """
        Registra un loader para un módulo.
        
        Args:
            name: Nombre del módulo
            loader: Función que carga el módulo
        """
        self._loaders[name] = loader
    
    def get(self, name: str) -> Any:
        """
        Obtiene módulo, cargándolo si es necesario.
        
        Args:
            name: Nombre del módulo
        
        Returns:
            Módulo cargado
        """
        if name not in self._modules:
            if name in self._loaders:
                self._modules[name] = self._loaders[name]()
            else:
                raise ValueError(f"Loader not registered for {name}")
        return self._modules[name]


class MemoryOptimizer:
    """Optimizador de memoria"""
    
    @staticmethod
    def optimize_memory():
        """Aplica optimizaciones de memoria"""
        # Forzar garbage collection
        gc.collect()
        
        # Configurar thresholds de GC
        gc.set_threshold(700, 10, 10)
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Obtiene uso de memoria"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,  # Resident Set Size
                "vms": memory_info.vms,  # Virtual Memory Size
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    @staticmethod
    def clear_cache():
        """Limpia caches para liberar memoria"""
        # Limpiar cache de functools.lru_cache
        # Esto requiere acceso a las funciones con cache
        pass


class WeakRefCache:
    """Cache usando weak references para no prevenir garbage collection"""
    
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        """Establece valor en cache"""
        self._cache[key] = value


def optimize_memory_usage():
    """Aplica optimizaciones de memoria globales"""
    MemoryOptimizer.optimize_memory()
    logger.info("Memory optimizations applied")


class StreamingResponse:
    """Helper para respuestas streaming que ahorran memoria"""
    
    @staticmethod
    async def stream_list(items: list, chunk_size: int = 100):
        """
        Stream de lista en chunks.
        
        Args:
            items: Lista de items
            chunk_size: Tamaño de chunk
        
        Yields:
            Chunks de items
        """
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]


def lazy_import(module_name: str):
    """
    Import lazy de un módulo.
    
    Args:
        module_name: Nombre del módulo
    
    Returns:
        Proxy que carga el módulo cuando se accede
    """
    class LazyModule:
        def __init__(self, name):
            self._name = name
            self._module = None
        
        def _load(self):
            if self._module is None:
                import importlib
                self._module = importlib.import_module(self._name)
            return self._module
        
        def __getattr__(self, name):
            return getattr(self._load(), name)
    
    return LazyModule(module_name)















