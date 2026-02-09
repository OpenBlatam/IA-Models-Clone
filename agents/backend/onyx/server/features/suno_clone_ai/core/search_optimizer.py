"""
Search Optimizer
Optimizaciones para búsqueda y recomendaciones
"""

import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)


class SearchOptimizer:
    """Optimizador para búsqueda"""
    
    def __init__(self):
        self._index_cache: Dict[str, Any] = {}
        self._vector_cache: Dict[str, np.ndarray] = {}
    
    @lru_cache(maxsize=256)
    def normalize_query(self, query: str) -> str:
        """
        Normaliza query para búsqueda
        
        Args:
            query: Query original
            
        Returns:
            Query normalizada
        """
        # Lowercase, trim, remove extra spaces
        normalized = " ".join(query.lower().strip().split())
        return normalized
    
    def create_search_index(self, items: List[Dict[str, Any]], key: str = "id") -> Dict[str, int]:
        """
        Crea índice de búsqueda
        
        Args:
            items: Lista de items
            key: Clave para indexar
            
        Returns:
            Índice
        """
        index = {item[key]: i for i, item in enumerate(items)}
        return index
    
    def vectorize_query(self, query: str, method: str = "tfidf") -> np.ndarray:
        """
        Vectoriza query para búsqueda semántica
        
        Args:
            query: Query de texto
            method: Método de vectorización
            
        Returns:
            Vector de la query
        """
        # Cache de vectores
        cache_key = f"{method}:{query}"
        if cache_key in self._vector_cache:
            return self._vector_cache[cache_key]
        
        # Vectorización simple (debería usar modelo real)
        # Por ahora, hash simple
        vector = np.zeros(128)  # Dimensión de ejemplo
        for i, char in enumerate(query[:128]):
            vector[i] = ord(char) / 255.0
        
        # Normalizar
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        self._vector_cache[cache_key] = vector
        return vector
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calcula similitud coseno optimizada
        
        Args:
            vec1: Vector 1
            vec2: Vector 2
            
        Returns:
            Similitud (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def batch_similarity(
        self,
        query_vector: np.ndarray,
        item_vectors: np.ndarray,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Calcula similitud en batch optimizado
        
        Args:
            query_vector: Vector de query
            item_vectors: Array de vectores de items
            top_k: Top K resultados
            
        Returns:
            Lista de (índice, similitud)
        """
        # Normalizar query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        
        query_normalized = query_vector / query_norm
        
        # Calcular similitudes en batch (vectorizado)
        similarities = np.dot(item_vectors, query_normalized)
        
        # Obtener top K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def optimize_search_params(
        self,
        total_items: int,
        query_complexity: str = "simple"
    ) -> Dict[str, Any]:
        """
        Optimiza parámetros de búsqueda
        
        Args:
            total_items: Total de items
            query_complexity: Complejidad de query
            
        Returns:
            Parámetros optimizados
        """
        if total_items < 1000:
            # Búsqueda lineal es suficiente
            return {
                "method": "linear",
                "batch_size": total_items,
                "use_index": False
            }
        elif total_items < 100000:
            # Usar índice
            return {
                "method": "indexed",
                "batch_size": 1000,
                "use_index": True
            }
        else:
            # Usar vectorización y aproximación
            return {
                "method": "vectorized",
                "batch_size": 10000,
                "use_index": True,
                "approximate": True
            }


# Instancia global
_search_optimizer: Optional[SearchOptimizer] = None


def get_search_optimizer() -> SearchOptimizer:
    """Obtiene el optimizador de búsqueda"""
    global _search_optimizer
    if _search_optimizer is None:
        _search_optimizer = SearchOptimizer()
    return _search_optimizer















