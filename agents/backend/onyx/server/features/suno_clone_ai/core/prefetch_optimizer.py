"""
Prefetch Optimizer
Pre-carga inteligente de datos para reducir latencia
"""

import logging
import asyncio
from typing import List, Any, Callable, Awaitable, Dict, Optional
from collections import deque
import time

logger = logging.getLogger(__name__)


class PrefetchOptimizer:
    """
    Optimizador de pre-carga
    Pre-carga datos que probablemente se necesitarán
    """
    
    def __init__(self, prefetch_window: int = 5):
        self.prefetch_window = prefetch_window
        self._prefetch_queue: deque = deque(maxlen=prefetch_window)
        self._prefetch_cache: Dict[str, Any] = {}
        self._prefetch_tasks: Dict[str, asyncio.Task] = {}
        self._access_patterns: Dict[str, List[float]] = {}
    
    def record_access(self, key: str):
        """Registra acceso a un recurso"""
        current_time = time.time()
        
        if key not in self._access_patterns:
            self._access_patterns[key] = []
        
        self._access_patterns[key].append(current_time)
        
        # Mantener solo últimos 100 accesos
        if len(self._access_patterns[key]) > 100:
            self._access_patterns[key] = self._access_patterns[key][-100:]
    
    def predict_next(self, current_key: str) -> List[str]:
        """
        Predice los siguientes recursos que se necesitarán
        
        Args:
            current_key: Recurso actual
            
        Returns:
            Lista de keys predichas
        """
        # Análisis simple de patrones de acceso
        predictions = []
        
        # Buscar patrones secuenciales
        for key, accesses in self._access_patterns.items():
            if key == current_key:
                continue
            
            # Si se accede frecuentemente después de current_key
            if len(accesses) > 5:
                predictions.append(key)
        
        return predictions[:self.prefetch_window]
    
    async def prefetch(self, keys: List[str], fetcher: Callable[[str], Awaitable[Any]]):
        """
        Pre-carga recursos
        
        Args:
            keys: Lista de keys a pre-cargar
            fetcher: Función async que obtiene el recurso
        """
        tasks = []
        for key in keys:
            if key not in self._prefetch_cache:
                task = asyncio.create_task(self._fetch_and_cache(key, fetcher))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _fetch_and_cache(self, key: str, fetcher: Callable[[str], Awaitable[Any]]):
        """Obtiene y cachea un recurso"""
        try:
            value = await fetcher(key)
            self._prefetch_cache[key] = {
                "value": value,
                "timestamp": time.time()
            }
            logger.debug(f"Prefetched: {key}")
        except Exception as e:
            logger.warning(f"Prefetch failed for {key}: {e}")
    
    def get_prefetched(self, key: str, max_age: float = 60.0) -> Optional[Any]:
        """
        Obtiene un recurso pre-cargado
        
        Args:
            key: Key del recurso
            max_age: Edad máxima en segundos
            
        Returns:
            Valor pre-cargado o None
        """
        if key in self._prefetch_cache:
            cached = self._prefetch_cache[key]
            age = time.time() - cached["timestamp"]
            if age < max_age:
                return cached["value"]
            else:
                del self._prefetch_cache[key]
        return None
    
    def clear_prefetch_cache(self):
        """Limpia el cache de pre-carga"""
        self._prefetch_cache.clear()


# Instancia global
_prefetch_optimizer: Optional[PrefetchOptimizer] = None


def get_prefetch_optimizer() -> PrefetchOptimizer:
    """Obtiene el optimizador de pre-carga"""
    global _prefetch_optimizer
    if _prefetch_optimizer is None:
        _prefetch_optimizer = PrefetchOptimizer()
    return _prefetch_optimizer















