"""
Servicio de Cache
================

Servicio para manejo de cache en memoria y persistente.
"""

import asyncio
import logging
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Entrada de cache"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at

@dataclass
class CacheStats:
    """Estadísticas de cache"""
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    hit_rate: float
    eviction_count: int
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]

class CacheService:
    """Servicio de cache"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl_seconds: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convertir a bytes
        self.default_ttl_seconds = default_ttl_seconds
        
        # Cache en memoria
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
        # Cache persistente
        self.persistent_cache_dir = Path("cache")
        self.persistent_cache_dir.mkdir(exist_ok=True)
        
        # Lock para operaciones thread-safe
        self._lock = threading.RLock()
        
        # Configuración
        self.enable_memory_cache = True
        self.enable_persistent_cache = True
        self.enable_compression = True
        
    async def initialize(self):
        """Inicializa el servicio de cache"""
        logger.info("Inicializando servicio de cache...")
        
        # Cargar cache persistente
        if self.enable_persistent_cache:
            await self._load_persistent_cache()
        
        # Iniciar limpieza automática
        asyncio.create_task(self._cleanup_worker())
        
        logger.info("Servicio de cache inicializado")
    
    async def _load_persistent_cache(self):
        """Carga cache persistente desde disco"""
        try:
            cache_files = list(self.persistent_cache_dir.glob("*.cache"))
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb') as f:
                        entry_data = pickle.load(f)
                    
                    # Verificar si la entrada no ha expirado
                    if entry_data.get('expires_at') is None or datetime.fromisoformat(entry_data['expires_at']) > datetime.now():
                        key = entry_data['key']
                        entry = CacheEntry(
                            key=key,
                            value=entry_data['value'],
                            created_at=datetime.fromisoformat(entry_data['created_at']),
                            expires_at=datetime.fromisoformat(entry_data['expires_at']) if entry_data.get('expires_at') else None,
                            access_count=entry_data.get('access_count', 0),
                            last_accessed=datetime.fromisoformat(entry_data.get('last_accessed', entry_data['created_at'])),
                            size_bytes=entry_data.get('size_bytes', 0)
                        )
                        
                        self.memory_cache[key] = entry
                    
                except Exception as e:
                    logger.warning(f"Error cargando entrada de cache {cache_file}: {e}")
                    # Eliminar archivo corrupto
                    cache_file.unlink(missing_ok=True)
            
            logger.info(f"Cache persistente cargado: {len(self.memory_cache)} entradas")
            
        except Exception as e:
            logger.error(f"Error cargando cache persistente: {e}")
    
    async def _save_persistent_cache(self, key: str, entry: CacheEntry):
        """Guarda entrada en cache persistente"""
        try:
            if not self.enable_persistent_cache:
                return
            
            cache_file = self.persistent_cache_dir / f"{self._hash_key(key)}.cache"
            
            entry_data = {
                'key': entry.key,
                'value': entry.value,
                'created_at': entry.created_at.isoformat(),
                'expires_at': entry.expires_at.isoformat() if entry.expires_at else None,
                'access_count': entry.access_count,
                'last_accessed': entry.last_accessed.isoformat(),
                'size_bytes': entry.size_bytes
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(entry_data, f)
            
        except Exception as e:
            logger.error(f"Error guardando entrada de cache: {e}")
    
    def _hash_key(self, key: str) -> str:
        """Genera hash de la clave"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _calculate_size(self, value: Any) -> int:
        """Calcula el tamaño aproximado de un valor"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float, bool)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) for k, v in value.items())
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Tamaño por defecto
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        try:
            with self._lock:
                # Verificar cache en memoria
                if self.enable_memory_cache and key in self.memory_cache:
                    entry = self.memory_cache[key]
                    
                    # Verificar expiración
                    if entry.expires_at and entry.expires_at <= datetime.now():
                        await self.delete(key)
                        self.cache_stats["misses"] += 1
                        return None
                    
                    # Actualizar estadísticas de acceso
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    
                    self.cache_stats["hits"] += 1
                    return entry.value
                
                # Verificar cache persistente
                if self.enable_persistent_cache:
                    cache_file = self.persistent_cache_dir / f"{self._hash_key(key)}.cache"
                    if cache_file.exists():
                        try:
                            with open(cache_file, 'rb') as f:
                                entry_data = pickle.load(f)
                            
                            # Verificar expiración
                            if entry_data.get('expires_at') is None or datetime.fromisoformat(entry_data['expires_at']) > datetime.now():
                                value = entry_data['value']
                                
                                # Cargar en memoria para acceso rápido
                                if self.enable_memory_cache:
                                    entry = CacheEntry(
                                        key=key,
                                        value=value,
                                        created_at=datetime.fromisoformat(entry_data['created_at']),
                                        expires_at=datetime.fromisoformat(entry_data['expires_at']) if entry_data.get('expires_at') else None,
                                        access_count=entry_data.get('access_count', 0) + 1,
                                        last_accessed=datetime.now(),
                                        size_bytes=entry_data.get('size_bytes', 0)
                                    )
                                    self.memory_cache[key] = entry
                                
                                self.cache_stats["hits"] += 1
                                return value
                            else:
                                # Eliminar entrada expirada
                                cache_file.unlink(missing_ok=True)
                        
                        except Exception as e:
                            logger.warning(f"Error cargando entrada persistente {key}: {e}")
                            cache_file.unlink(missing_ok=True)
                
                self.cache_stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error obteniendo del cache: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Establece valor en cache"""
        try:
            with self._lock:
                ttl = ttl_seconds or self.default_ttl_seconds
                expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
                
                # Calcular tamaño
                size_bytes = self._calculate_size(value)
                
                # Crear entrada
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    expires_at=expires_at,
                    access_count=0,
                    last_accessed=datetime.now(),
                    size_bytes=size_bytes
                )
                
                # Verificar espacio disponible
                await self._ensure_space_available(size_bytes)
                
                # Guardar en memoria
                if self.enable_memory_cache:
                    self.memory_cache[key] = entry
                
                # Guardar en persistente
                if self.enable_persistent_cache:
                    await self._save_persistent_cache(key, entry)
                
                return True
                
        except Exception as e:
            logger.error(f"Error estableciendo en cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Elimina entrada del cache"""
        try:
            with self._lock:
                deleted = False
                
                # Eliminar de memoria
                if self.enable_memory_cache and key in self.memory_cache:
                    del self.memory_cache[key]
                    deleted = True
                
                # Eliminar de persistente
                if self.enable_persistent_cache:
                    cache_file = self.persistent_cache_dir / f"{self._hash_key(key)}.cache"
                    if cache_file.exists():
                        cache_file.unlink()
                        deleted = True
                
                return deleted
                
        except Exception as e:
            logger.error(f"Error eliminando del cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """Limpia todo el cache"""
        try:
            with self._lock:
                # Limpiar memoria
                if self.enable_memory_cache:
                    self.memory_cache.clear()
                
                # Limpiar persistente
                if self.enable_persistent_cache:
                    cache_files = list(self.persistent_cache_dir.glob("*.cache"))
                    for cache_file in cache_files:
                        cache_file.unlink(missing_ok=True)
                
                # Resetear estadísticas
                self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
                
                return True
                
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}")
            return False
    
    async def _ensure_space_available(self, required_size: int):
        """Asegura que hay espacio disponible en el cache"""
        try:
            current_size = sum(entry.size_bytes for entry in self.memory_cache.values())
            
            # Si necesitamos más espacio, evictar entradas
            while current_size + required_size > self.max_size_bytes and self.memory_cache:
                await self._evict_least_recently_used()
                current_size = sum(entry.size_bytes for entry in self.memory_cache.values())
                
        except Exception as e:
            logger.error(f"Error asegurando espacio en cache: {e}")
    
    async def _evict_least_recently_used(self):
        """Evicta la entrada menos recientemente usada"""
        try:
            if not self.memory_cache:
                return
            
            # Encontrar entrada con menor last_accessed
            lru_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].last_accessed
            )
            
            # Eliminar entrada
            del self.memory_cache[lru_key]
            self.cache_stats["evictions"] += 1
            
            # Eliminar también del cache persistente
            if self.enable_persistent_cache:
                cache_file = self.persistent_cache_dir / f"{self._hash_key(lru_key)}.cache"
                cache_file.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Error evictando entrada LRU: {e}")
    
    async def _cleanup_worker(self):
        """Worker que limpia entradas expiradas"""
        while True:
            try:
                await asyncio.sleep(300)  # Ejecutar cada 5 minutos
                await self._cleanup_expired_entries()
            except Exception as e:
                logger.error(f"Error en cleanup worker: {e}")
    
    async def _cleanup_expired_entries(self):
        """Limpia entradas expiradas"""
        try:
            with self._lock:
                now = datetime.now()
                expired_keys = []
                
                # Encontrar entradas expiradas en memoria
                for key, entry in self.memory_cache.items():
                    if entry.expires_at and entry.expires_at <= now:
                        expired_keys.append(key)
                
                # Eliminar entradas expiradas
                for key in expired_keys:
                    await self.delete(key)
                
                if expired_keys:
                    logger.info(f"Limpiadas {len(expired_keys)} entradas expiradas del cache")
                
        except Exception as e:
            logger.error(f"Error limpiando entradas expiradas: {e}")
    
    async def get_or_set(self, key: str, factory: Callable, ttl_seconds: Optional[int] = None) -> Any:
        """Obtiene valor del cache o lo genera usando factory"""
        try:
            # Intentar obtener del cache
            value = await self.get(key)
            if value is not None:
                return value
            
            # Generar valor usando factory
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()
            
            # Guardar en cache
            await self.set(key, value, ttl_seconds)
            
            return value
            
        except Exception as e:
            logger.error(f"Error en get_or_set: {e}")
            # Intentar ejecutar factory como fallback
            try:
                if asyncio.iscoroutinefunction(factory):
                    return await factory()
                else:
                    return factory()
            except Exception:
                raise e
    
    async def get_stats(self) -> CacheStats:
        """Obtiene estadísticas del cache"""
        try:
            with self._lock:
                total_hits = self.cache_stats["hits"]
                total_misses = self.cache_stats["misses"]
                total_requests = total_hits + total_misses
                hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
                
                total_size = sum(entry.size_bytes for entry in self.memory_cache.values())
                
                oldest_entry = min(
                    (entry.created_at for entry in self.memory_cache.values()),
                    default=None
                )
                newest_entry = max(
                    (entry.created_at for entry in self.memory_cache.values()),
                    default=None
                )
                
                return CacheStats(
                    total_entries=len(self.memory_cache),
                    total_size_bytes=total_size,
                    hit_count=total_hits,
                    miss_count=total_misses,
                    hit_rate=hit_rate,
                    eviction_count=self.cache_stats["evictions"],
                    oldest_entry=oldest_entry,
                    newest_entry=newest_entry
                )
                
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de cache: {e}")
            return CacheStats(0, 0, 0, 0, 0.0, 0, None, None)
    
    async def warm_cache(self, keys_and_factories: Dict[str, Callable], ttl_seconds: Optional[int] = None):
        """Pre-carga cache con valores"""
        try:
            tasks = []
            for key, factory in keys_and_factories.items():
                task = self.get_or_set(key, factory, ttl_seconds)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            logger.info(f"Cache pre-cargado con {len(keys_and_factories)} entradas")
            
        except Exception as e:
            logger.error(f"Error pre-cargando cache: {e}")
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalida entradas que coincidan con un patrón"""
        try:
            import fnmatch
            
            with self._lock:
                keys_to_delete = []
                
                for key in self.memory_cache.keys():
                    if fnmatch.fnmatch(key, pattern):
                        keys_to_delete.append(key)
                
                # Eliminar entradas
                for key in keys_to_delete:
                    await self.delete(key)
                
                return len(keys_to_delete)
                
        except Exception as e:
            logger.error(f"Error invalidando patrón {pattern}: {e}")
            return 0


