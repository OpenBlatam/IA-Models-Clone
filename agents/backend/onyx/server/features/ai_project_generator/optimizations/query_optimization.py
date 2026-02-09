"""
Query Optimization - Optimización de queries
===========================================

Optimizaciones para queries y búsquedas.
"""

import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Optimizador de queries"""
    
    @staticmethod
    def optimize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimiza filtros de query.
        
        Args:
            filters: Filtros originales
        
        Returns:
            Filtros optimizados
        """
        # Remover filtros vacíos
        optimized = {
            k: v for k, v in filters.items()
            if v is not None and v != "" and v != []
        }
        
        # Ordenar filtros por selectividad (más selectivos primero)
        # Esto es una simplificación, en producción usar estadísticas
        priority_order = ["id", "project_id", "status", "author", "created_at"]
        sorted_filters = {}
        for key in priority_order:
            if key in optimized:
                sorted_filters[key] = optimized.pop(key)
        sorted_filters.update(optimized)
        
        return sorted_filters
    
    @staticmethod
    def optimize_pagination(limit: int, offset: int, max_limit: int = 1000) -> tuple:
        """
        Optimiza paginación.
        
        Args:
            limit: Límite
            offset: Offset
            max_limit: Límite máximo
        
        Returns:
            (limit, offset) optimizados
        """
        limit = min(max(limit, 1), max_limit)
        offset = max(offset, 0)
        return limit, offset
    
    @staticmethod
    def build_index_hint(filters: Dict[str, Any]) -> Optional[str]:
        """
        Construye hint de índice basado en filtros.
        
        Args:
            filters: Filtros
        
        Returns:
            Hint de índice o None
        """
        # En producción, esto usaría estadísticas de la base de datos
        if "project_id" in filters:
            return "idx_project_id"
        elif "status" in filters and "author" in filters:
            return "idx_status_author"
        elif "status" in filters:
            return "idx_status"
        return None


class QueryCache:
    """Cache de queries optimizado"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
    
    def get_cache_key(self, query: str, filters: Dict[str, Any]) -> str:
        """Genera clave de cache para query"""
        import hashlib
        import json
        key_data = {
            "query": query,
            "filters": json.dumps(filters, sort_keys=True)
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene resultado de cache"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        """Establece resultado en cache"""
        if len(self._cache) >= self.max_size:
            # LRU: eliminar el más antiguo
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value


class BatchQueryOptimizer:
    """Optimizador de queries en batch"""
    
    @staticmethod
    async def batch_get(
        repository,
        ids: List[str],
        batch_size: int = 100
    ) -> List[Any]:
        """
        Obtiene múltiples items en batch.
        
        Args:
            repository: Repositorio
            ids: Lista de IDs
            batch_size: Tamaño de batch
        
        Returns:
            Lista de items
        """
        results = []
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[repository.get_by_id(id) for id in batch],
                return_exceptions=True
            )
            results.extend([
                r for r in batch_results
                if not isinstance(r, Exception) and r is not None
            ])
        return results


# Importar asyncio
import asyncio















