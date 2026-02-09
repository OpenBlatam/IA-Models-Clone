"""
Cache - Sistema de caché para optimizar operaciones repetitivas
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


class LRUCache:
    """Cache LRU (Least Recently Used) simple"""

    def __init__(self, max_size: int = 100):
        """
        Inicializar cache LRU.

        Args:
            max_size: Tamaño máximo del cache
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del cache.

        Args:
            key: Clave del cache

        Returns:
            Valor almacenado o None
        """
        if key in self.cache:
            # Mover al final (más reciente)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """
        Almacenar valor en cache.

        Args:
            key: Clave del cache
            value: Valor a almacenar
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Eliminar el menos reciente (primero)
                self.cache.popitem(last=False)
        
        self.cache[key] = value

    def clear(self):
        """Limpiar todo el cache"""
        self.cache.clear()

    def size(self) -> int:
        """Obtener tamaño actual del cache"""
        return len(self.cache)


class AnalysisCache:
    """Cache para análisis de contenido"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Inicializar cache de análisis.

        Args:
            max_size: Tamaño máximo del cache
            ttl_seconds: Tiempo de vida en segundos
        """
        self.cache = LRUCache(max_size)
        self.ttl = timedelta(seconds=ttl_seconds)

    def _generate_key(self, content: str, operation: str = "analyze") -> str:
        """
        Generar clave única para el contenido.

        Args:
            content: Contenido a analizar
            operation: Tipo de operación

        Returns:
            Clave hash
        """
        # Usar hash del contenido para la clave
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{operation}:{content_hash}"

    def get_analysis(self, content: str, operation: str = "analyze") -> Optional[Dict[str, Any]]:
        """
        Obtener análisis del cache.

        Args:
            content: Contenido analizado
            operation: Tipo de operación

        Returns:
            Análisis cacheado o None
        """
        key = self._generate_key(content, operation)
        cached = self.cache.get(key)
        
        if cached:
            # Verificar TTL
            timestamp = cached.get("timestamp")
            if timestamp:
                cache_time = datetime.fromisoformat(timestamp)
                if datetime.utcnow() - cache_time < self.ttl:
                    return cached.get("data")
                else:
                    # Expiró, eliminar
                    self.cache.cache.pop(key, None)
        
        return None

    def set_analysis(
        self,
        content: str,
        analysis: Dict[str, Any],
        operation: str = "analyze"
    ):
        """
        Almacenar análisis en cache.

        Args:
            content: Contenido analizado
            analysis: Resultado del análisis
            operation: Tipo de operación
        """
        key = self._generate_key(content, operation)
        self.cache.set(key, {
            "data": analysis,
            "timestamp": datetime.utcnow().isoformat()
        })

    def clear(self):
        """Limpiar todo el cache"""
        self.cache.clear()






