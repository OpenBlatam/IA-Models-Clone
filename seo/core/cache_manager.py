from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
from typing import Dict, List, Any, Optional
from loguru import logger
import orjson
import zstandard
from cachetools import TTLCache
from .interfaces import CacheManager
            import diskcache
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Cache Manager ultra-optimizado para el servicio SEO.
Implementación con compresión Zstandard y multi-nivel.
"""




class UltraOptimizedCacheManager(CacheManager):
    """Gestor de caché ultra-optimizado con compresión."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.cache = TTLCache(
            maxsize=self.config.get('maxsize', 2000),
            ttl=self.config.get('ttl', 3600)
        )
        
        # Configurar compresión
        compression_level = self.config.get('compression_level', 3)
        self.compressor = zstandard.ZstdCompressor(level=compression_level)
        self.decompressor = zstandard.ZstdDecompressor()
        
        # Estadísticas
        self.stats = {
            "hits": 0,
            "misses": 0,
            "compression_ratio": 0.0,
            "total_compressed_size": 0,
            "total_original_size": 0,
            "evictions": 0
        }
        
        # Callback para evictions
        self.cache._on_evict = self._on_evict
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene datos del caché con descompresión."""
        try:
            compressed_data = self.cache.get(key)
            if compressed_data:
                self.stats["hits"] += 1
                decompressed_data = self.decompressor.decompress(compressed_data)
                return orjson.loads(decompressed_data)
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any):
        """Almacena datos en caché con compresión."""
        try:
            json_data = orjson.dumps(value)
            compressed_data = self.compressor.compress(json_data)
            
            # Actualizar estadísticas de compresión
            original_size = len(json_data)
            compressed_size = len(compressed_data)
            compression_ratio = (original_size - compressed_size) / original_size
            
            self.stats["compression_ratio"] = compression_ratio
            self.stats["total_compressed_size"] += compressed_size
            self.stats["total_original_size"] += original_size
            
            self.cache[key] = compressed_data
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
    
    def set_many(self, data: Dict[str, Any]):
        """Almacena múltiples elementos en caché."""
        for key, value in data.items():
            self.set(key, value)
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Obtiene múltiples elementos del caché."""
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def delete(self, key: str) -> bool:
        """Elimina un elemento del caché."""
        try:
            return key in self.cache and self.cache.pop(key, None) is not None
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear(self) -> int:
        """Limpia el caché y retorna elementos eliminados."""
        size = len(self.cache)
        self.cache.clear()
        return size
    
    def exists(self, key: str) -> bool:
        """Verifica si una clave existe en el caché."""
        return key in self.cache
    
    def keys(self) -> List[str]:
        """Obtiene todas las claves del caché."""
        return list(self.cache.keys())
    
    def size(self) -> int:
        """Obtiene el tamaño actual del caché."""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del caché."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        avg_compression_ratio = (
            self.stats["compression_ratio"] if self.stats["total_original_size"] > 0 else 0
        )
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "compression_ratio": avg_compression_ratio,
            "cache_size": len(self.cache),
            "max_size": self.cache.maxsize,
            "ttl": self.cache.ttl,
            "evictions": self.stats["evictions"],
            "total_compressed_size_mb": self.stats["total_compressed_size"] / 1024 / 1024,
            "total_original_size_mb": self.stats["total_original_size"] / 1024 / 1024,
            "memory_saved_mb": (
                self.stats["total_original_size"] - self.stats["total_compressed_size"]
            ) / 1024 / 1024
        }
    
    def _on_evict(self, key, value) -> Any:
        """Callback cuando se evicta un elemento del caché."""
        self.stats["evictions"] += 1
    
    def warm_up(self, data: Dict[str, Any]):
        """Pre-carga datos en el caché."""
        logger.info(f"Warming up cache with {len(data)} items")
        self.set_many(data)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Obtiene métricas de rendimiento del caché."""
        stats = self.get_stats()
        return {
            "hit_rate": stats["hit_rate"],
            "compression_ratio": stats["compression_ratio"],
            "memory_efficiency": stats["memory_saved_mb"] / max(stats["total_original_size_mb"], 1),
            "eviction_rate": stats["evictions"] / max(stats["hits"] + stats["misses"], 1)
        }


class MultiLevelCacheManager(CacheManager):
    """Gestor de caché multi-nivel ultra-optimizado."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        
        # L1: Memory cache (más rápido)
        self.l1_cache = UltraOptimizedCacheManager({
            'maxsize': self.config.get('l1_maxsize', 1000),
            'ttl': self.config.get('l1_ttl', 300),  # 5 minutos
            'compression_level': 1  # Compresión rápida
        })
        
        # L2: Disk cache (más lento pero persistente)
        self.l2_cache = None
        if self.config.get('enable_l2', True):
            self.l2_cache = self._create_l2_cache()
    
    def _create_l2_cache(self) -> Any:
        """Crea caché de nivel 2 (disco)."""
        try:
            return diskcache.Cache(
                directory=self.config.get('l2_directory', './cache'),
                size_limit=self.config.get('l2_size_limit', 1024 * 1024 * 100),  # 100MB
                timeout=self.config.get('l2_timeout', 3600)
            )
        except ImportError:
            logger.warning("DiskCache not available, L2 cache disabled")
            return None
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene datos del caché multi-nivel."""
        # Intentar L1 primero
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Intentar L2 si está disponible
        if self.l2_cache:
            try:
                value = self.l2_cache.get(key)
                if value is not None:
                    # Promover a L1
                    self.l1_cache.set(key, value)
                    return value
            except Exception as e:
                logger.error(f"L2 cache get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any):
        """Almacena datos en ambos niveles de caché."""
        # L1 cache
        self.l1_cache.set(key, value)
        
        # L2 cache si está disponible
        if self.l2_cache:
            try:
                self.l2_cache.set(key, value)
            except Exception as e:
                logger.error(f"L2 cache set error: {e}")
    
    def clear(self) -> int:
        """Limpia ambos niveles de caché."""
        l1_cleared = self.l1_cache.clear()
        l2_cleared = 0
        
        if self.l2_cache:
            try:
                l2_cleared = len(self.l2_cache)
                self.l2_cache.clear()
            except Exception as e:
                logger.error(f"L2 cache clear error: {e}")
        
        return l1_cleared + l2_cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas combinadas de ambos niveles."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = {}
        
        if self.l2_cache:
            try:
                l2_stats = {
                    "l2_size": len(self.l2_cache),
                    "l2_directory": self.config.get('l2_directory', './cache')
                }
            except Exception as e:
                logger.error(f"L2 cache stats error: {e}")
        
        return {
            "l1": l1_stats,
            "l2": l2_stats,
            "combined": {
                "total_hits": l1_stats["hits"],
                "total_misses": l1_stats["misses"],
                "overall_hit_rate": l1_stats["hit_rate"],
                "total_size": l1_stats["cache_size"] + l2_stats.get("l2_size", 0)
            }
        }


class CacheManagerFactory:
    """Factory para crear gestores de caché."""
    
    @staticmethod
    def create_cache_manager(
        cache_type: str = "ultra_optimized", 
        config: Optional[Dict[str, Any]] = None
    ) -> CacheManager:
        """Crea un gestor de caché basado en el tipo especificado."""
        if cache_type == "ultra_optimized":
            return UltraOptimizedCacheManager(config)
        elif cache_type == "multi_level":
            return MultiLevelCacheManager(config)
        else:
            raise ValueError(f"Unknown cache manager type: {cache_type}") 