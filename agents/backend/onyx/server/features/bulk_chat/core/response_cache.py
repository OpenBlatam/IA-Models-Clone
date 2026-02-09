"""
Response Cache - Cache de respuestas
=====================================

Sistema de cache para respuestas de LLM, reduciendo llamadas duplicadas
y mejorando el rendimiento.
"""

import hashlib
import json
import asyncio
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Respuesta en cache."""
    response: str
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseCache:
    """
    Cache de respuestas con LRU (Least Recently Used) eviction.
    
    Características:
    - Cache basado en hash del mensaje
    - TTL configurable
    - LRU eviction automático
    - Estadísticas de hit/miss
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,  # 1 hora
        enable_semantic_cache: bool = False,
    ):
        """
        Inicializar cache.
        
        Args:
            max_size: Tamaño máximo del cache
            ttl_seconds: Tiempo de vida en segundos
            enable_semantic_cache: Habilitar cache semántico (requiere embeddings)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_semantic_cache = enable_semantic_cache
        
        # Cache: {hash: CachedResponse}
        self.cache: OrderedDict[str, CachedResponse] = OrderedDict()
        
        # Estadísticas
        self.hits = 0
        self.misses = 0
        
        # Lock para thread safety
        self._lock = asyncio.Lock()
    
    def _hash_message(self, messages: list) -> str:
        """Generar hash del mensaje."""
        # Convertir mensajes a string consistente
        message_str = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(message_str.encode("utf-8")).hexdigest()
    
    async def get(
        self,
        messages: list,
        similarity_threshold: float = 0.9,
    ) -> Optional[str]:
        """
        Obtener respuesta del cache.
        
        Args:
            messages: Lista de mensajes
            similarity_threshold: Umbral de similitud para cache semántico
        
        Returns:
            Respuesta en cache o None
        """
        async with self._lock:
            # Limpiar entradas expiradas
            await self._cleanup_expired()
            
            # Buscar por hash exacto
            cache_key = self._hash_message(messages)
            
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                
                # Verificar si está expirado
                if datetime.now() - cached.created_at > timedelta(seconds=self.ttl_seconds):
                    del self.cache[cache_key]
                    self.misses += 1
                    return None
                
                # Actualizar estadísticas
                cached.access_count += 1
                cached.last_accessed = datetime.now()
                
                # Mover al final (LRU)
                self.cache.move_to_end(cache_key)
                
                self.hits += 1
                logger.debug(f"Cache HIT for key: {cache_key[:8]}...")
                return cached.response
            
            # Si está habilitado, buscar por similitud semántica
            if self.enable_semantic_cache:
                similar_response = await self._find_similar(messages, similarity_threshold)
                if similar_response:
                    self.hits += 1
                    return similar_response
            
            self.misses += 1
            logger.debug(f"Cache MISS for key: {cache_key[:8]}...")
            return None
    
    async def set(
        self,
        messages: list,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Guardar respuesta en cache.
        
        Args:
            messages: Lista de mensajes
            response: Respuesta a cachear
            metadata: Metadatos adicionales
        """
        async with self._lock:
            # Limpiar si está lleno (LRU eviction)
            if len(self.cache) >= self.max_size:
                # Eliminar el menos recientemente usado
                self.cache.popitem(last=False)
            
            cache_key = self._hash_message(messages)
            
            self.cache[cache_key] = CachedResponse(
                response=response,
                created_at=datetime.now(),
                metadata=metadata or {},
            )
            
            # Mover al final
            self.cache.move_to_end(cache_key)
            
            logger.debug(f"Cached response for key: {cache_key[:8]}...")
    
    async def _cleanup_expired(self):
        """Limpiar entradas expiradas."""
        now = datetime.now()
        expired_keys = [
            key
            for key, cached in self.cache.items()
            if now - cached.created_at > timedelta(seconds=self.ttl_seconds)
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _find_similar(
        self,
        messages: list,
        threshold: float,
    ) -> Optional[str]:
        """
        Buscar respuesta similar usando embeddings.
        
        Requiere embeddings para funcionar.
        """
        # TODO: Implementar búsqueda semántica con embeddings
        # Por ahora, retornar None
        return None
    
    async def clear(self):
        """Limpiar todo el cache."""
        async with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }
    
    async def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidar entradas del cache.
        
        Args:
            pattern: Patrón para invalidar (si None, limpia todo)
        """
        async with self._lock:
            if pattern is None:
                await self.clear()
                return
            
            # Invalidar por patrón (simplificado)
            keys_to_remove = [
                key
                for key in self.cache.keys()
                if pattern in key
            ]
            
            for key in keys_to_remove:
                del self.cache[key]
            
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
































