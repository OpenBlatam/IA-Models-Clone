"""
Caché de Embeddings
===================

Caché en memoria para embeddings.
"""

import logging
import hashlib
import numpy as np
from typing import Dict, Optional, List, Union
from functools import lru_cache

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Caché LRU para embeddings."""
    
    def __init__(self, max_size: int = 1000):
        """
        Inicializar caché.
        
        Args:
            max_size: Tamaño máximo del caché
        """
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0
    
    def _get_key(self, text: str) -> str:
        """Generar clave para texto."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Obtener embedding del caché.
        
        Args:
            text: Texto
        
        Returns:
            Embedding o None
        """
        key = self._get_key(text)
        
        if key in self._cache:
            # Mover al final (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            self._hits += 1
            return self._cache[key].copy()
        
        self._misses += 1
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """
        Guardar embedding en caché.
        
        Args:
            text: Texto
            embedding: Embedding
        """
        key = self._get_key(text)
        
        # Si está lleno, eliminar el más antiguo
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = embedding.copy()
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def get_batch(
        self,
        texts: List[str]
    ) -> tuple[List[Optional[np.ndarray]], List[str]]:
        """
        Obtener embeddings en batch.
        
        Args:
            texts: Lista de textos
        
        Returns:
            (embeddings, textos_no_encontrados)
        """
        embeddings = []
        not_found = []
        
        for i, text in enumerate(texts):
            emb = self.get(text)
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(None)
                not_found.append(text)
        
        return embeddings, not_found
    
    def set_batch(self, texts: List[str], embeddings: List[np.ndarray]):
        """Guardar embeddings en batch."""
        for text, emb in zip(texts, embeddings):
            self.set(text, emb)
    
    def clear(self):
        """Limpiar caché."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_requests": total
        }




