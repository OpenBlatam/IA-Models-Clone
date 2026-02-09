"""
Memory Optimizer
Optimizaciones de memoria
"""

import logging
import gc
import sys
from typing import Dict, Any, Optional
import weakref

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Optimizador de memoria"""
    
    def __init__(self):
        self._weak_refs: Dict[str, weakref.ref] = {}
        self._memory_threshold = 0.8  # 80% de memoria
        self._gc_thresholds = (700, 10, 10)
    
    def optimize_gc(self):
        """Optimiza garbage collection"""
        # Ajustar thresholds para menos collections frecuentes
        gc.set_threshold(*self._gc_thresholds)
        logger.info("GC thresholds optimized")
    
    def enable_aggressive_gc(self):
        """Habilita GC agresivo (útil en Lambda)"""
        gc.set_threshold(100, 5, 5)
        logger.info("Aggressive GC enabled")
    
    def clear_weak_refs(self):
        """Limpia referencias débiles"""
        to_remove = []
        for key, ref in self._weak_refs.items():
            if ref() is None:
                to_remove.append(key)
        
        for key in to_remove:
            del self._weak_refs[key]
    
    def register_weak_ref(self, name: str, obj: Any):
        """Registra una referencia débil"""
        self._weak_refs[name] = weakref.ref(obj)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Obtiene uso de memoria"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def force_gc(self):
        """Fuerza garbage collection"""
        collected = gc.collect()
        logger.debug(f"GC collected {collected} objects")
        return collected
    
    def optimize_imports_memory(self):
        """Optimiza memoria de imports"""
        # Limpiar módulos no usados
        modules_to_keep = {
            'sys', 'os', 'logging', 'asyncio', 'json',
            'datetime', 'typing', 'functools', 'collections'
        }
        
        # Esto es agresivo, solo para entornos con memoria limitada
        # No hacer en desarrollo


# Instancia global
_memory_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Obtiene el optimizador de memoria"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer















