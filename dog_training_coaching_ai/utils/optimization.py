"""
Optimization Utilities
======================
Utilidades para optimización.
"""

from typing import List, Callable, Any, Dict, Optional, Tuple
import asyncio
from functools import lru_cache
import time

from .logger import get_logger

logger = get_logger(__name__)


class Memoizer:
    """Memoizer para cachear resultados de funciones."""
    
    def __init__(self, max_size: int = 128):
        """
        Inicializar memoizer.
        
        Args:
            max_size: Tamaño máximo del cache
        """
        self.cache: Dict[Tuple, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def memoize(self, func: Callable) -> Callable:
        """
        Decorator para memoizar función.
        
        Args:
            func: Función a memoizar
            
        Returns:
            Función memoizada
        """
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            result = func(*args, **kwargs)
            
            # LRU: eliminar más antiguo si está lleno
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = result
            return result
        
        return wrapper
    
    def clear(self):
        """Limpiar cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total
        }


class BatchProcessor:
    """Procesador de lotes optimizado."""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        """
        Inicializar batch processor.
        
        Args:
            batch_size: Tamaño del lote
            max_workers: Número máximo de workers
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    async def process_batches(
        self,
        items: List[Any],
        processor: Callable[[List[Any]], Any],
        parallel: bool = True
    ) -> List[Any]:
        """
        Procesar items en lotes.
        
        Args:
            items: Items a procesar
            processor: Función de procesamiento
            parallel: Procesar en paralelo
            
        Returns:
            Resultados
        """
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        if parallel:
            tasks = [processor(batch) for batch in batches]
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for batch in batches:
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(batch)
                else:
                    result = processor(batch)
                results.append(result)
        
        # Aplanar resultados si es necesario
        if results and isinstance(results[0], list):
            return [item for sublist in results for item in sublist]
        
        return results


def optimize_query_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimizar parámetros de query.
    
    Args:
        params: Parámetros originales
        
    Returns:
        Parámetros optimizados
    """
    optimized = {}
    
    for key, value in params.items():
        # Eliminar valores None
        if value is None:
            continue
        
        # Eliminar strings vacíos
        if isinstance(value, str) and not value.strip():
            continue
        
        # Eliminar listas vacías
        if isinstance(value, list) and len(value) == 0:
            continue
        
        optimized[key] = value
    
    return optimized


def deduplicate_list(items: List[Any], key_func: Optional[Callable] = None) -> List[Any]:
    """
    Eliminar duplicados de lista.
    
    Args:
        items: Items
        key_func: Función para extraer clave de comparación
        
    Returns:
        Lista sin duplicados
    """
    if key_func is None:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    
    seen = set()
    result = []
    for item in items:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


class LazyLoader:
    """Cargador lazy para recursos pesados."""
    
    def __init__(self, loader_func: Callable[[], Any]):
        """
        Inicializar lazy loader.
        
        Args:
            loader_func: Función de carga
        """
        self.loader_func = loader_func
        self._value: Optional[Any] = None
        self._loaded = False
    
    @property
    def value(self) -> Any:
        """Obtener valor (carga lazy)."""
        if not self._loaded:
            self._value = self.loader_func()
            self._loaded = True
        return self._value
    
    def reset(self):
        """Resetear loader."""
        self._value = None
        self._loaded = False


def debounce(func: Callable, delay: float = 0.5):
    """
    Debounce de función.
    
    Args:
        func: Función a debouncear
        delay: Delay en segundos
        
    Returns:
        Función debounceada
    """
    last_call = [0]
    
    def wrapper(*args, **kwargs):
        now = time.time()
        if now - last_call[0] >= delay:
            last_call[0] = now
            return func(*args, **kwargs)
    
    return wrapper


def throttle(func: Callable, limit: float = 1.0):
    """
    Throttle de función.
    
    Args:
        func: Función a throttlear
        limit: Límite de tiempo entre llamadas
        
    Returns:
        Función throttleada
    """
    last_call = [0]
    
    def wrapper(*args, **kwargs):
        now = time.time()
        if now - last_call[0] >= limit:
            last_call[0] = now
            return func(*args, **kwargs)
    
    return wrapper



