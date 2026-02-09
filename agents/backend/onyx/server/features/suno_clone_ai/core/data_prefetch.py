"""
Data Prefetch
Pre-carga inteligente de datos basada en patrones
"""

import logging
import asyncio
from typing import Dict, Any, List, Callable, Awaitable, Optional
from collections import defaultdict, deque, Counter
import time

logger = logging.getLogger(__name__)


class DataPrefetcher:
    """Pre-cargador inteligente de datos"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self._access_patterns: Dict[str, List[str]] = defaultdict(list)
        self._prefetch_cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, float] = {}
        self._prefetch_tasks: Dict[str, asyncio.Task] = {}
    
    def record_sequence(self, sequence: List[str]):
        """
        Registra una secuencia de accesos
        
        Args:
            sequence: Secuencia de keys accedidas
        """
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_key = sequence[i + 1]
            self._access_patterns[current].append(next_key)
    
    def predict_next(self, current_key: str, top_n: int = 5) -> List[str]:
        """
        Predice los siguientes keys basado en patrones
        
        Args:
            current_key: Key actual
            top_n: Número de predicciones
            
        Returns:
            Lista de keys predichas
        """
        if current_key not in self._access_patterns:
            return []
        
        # Contar frecuencias
        next_keys = self._access_patterns[current_key]
        if not next_keys:
            return []
        
        # Contar ocurrencias
        counter = Counter(next_keys)
        
        # Retornar top N
        return [key for key, _ in counter.most_common(top_n)]
    
    async def prefetch(
        self,
        keys: List[str],
        fetcher: Callable[[str], Awaitable[Any]],
        background: bool = True
    ):
        """
        Pre-carga keys
        
        Args:
            keys: Lista de keys a pre-cargar
            fetcher: Función async que obtiene el dato
            background: Si ejecutar en background
        """
        async def fetch_all():
            tasks = []
            for key in keys:
                if key not in self._prefetch_cache:
                    tasks.append(self._fetch_and_cache(key, fetcher))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        if background:
            asyncio.create_task(fetch_all())
        else:
            await fetch_all()
    
    async def _fetch_and_cache(self, key: str, fetcher: Callable[[str], Awaitable[Any]]):
        """Obtiene y cachea un dato"""
        try:
            value = await fetcher(key)
            
            # Limpiar cache si es muy grande
            if len(self._prefetch_cache) >= self.max_cache_size:
                self._evict_oldest()
            
            self._prefetch_cache[key] = value
            self._cache_times[key] = time.time()
            logger.debug(f"Prefetched: {key}")
        except Exception as e:
            logger.warning(f"Prefetch failed for {key}: {e}")
    
    def _evict_oldest(self):
        """Elimina la entrada más antigua"""
        if not self._cache_times:
            return
        
        oldest_key = min(self._cache_times.items(), key=lambda x: x[1])[0]
        del self._prefetch_cache[oldest_key]
        del self._cache_times[oldest_key]
    
    def get_prefetched(self, key: str, max_age: float = 300.0) -> Optional[Any]:
        """
        Obtiene dato pre-cargado
        
        Args:
            key: Key del dato
            max_age: Edad máxima en segundos
            
        Returns:
            Dato o None
        """
        if key in self._prefetch_cache:
            age = time.time() - self._cache_times.get(key, 0)
            if age < max_age:
                return self._prefetch_cache[key]
            else:
                del self._prefetch_cache[key]
                if key in self._cache_times:
                    del self._cache_times[key]
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        return {
            "cache_size": len(self._prefetch_cache),
            "patterns_tracked": len(self._access_patterns),
            "total_patterns": sum(len(v) for v in self._access_patterns.values())
        }


# Instancia global
_data_prefetcher: Optional[DataPrefetcher] = None


def get_data_prefetcher() -> DataPrefetcher:
    """Obtiene el pre-cargador de datos"""
    global _data_prefetcher
    if _data_prefetcher is None:
        _data_prefetcher = DataPrefetcher()
    return _data_prefetcher

