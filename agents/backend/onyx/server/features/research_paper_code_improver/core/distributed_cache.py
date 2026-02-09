"""
Distributed Cache - Sistema de cache distribuido (Redis-like)
==============================================================
"""

import logging
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entrada de cache"""
    key: str
    value: Any
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def is_expired(self) -> bool:
        """Verifica si expiró"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "key": self.key,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
        }


class DistributedCache:
    """Sistema de cache distribuido"""
    
    def __init__(self, max_size: int = 10000, default_ttl: Optional[int] = None):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl  # segundos
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    def _serialize(self, value: Any) -> bytes:
        """Serializa un valor"""
        try:
            return pickle.dumps(value)
        except Exception:
            return json.dumps(value).encode()
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserializa un valor"""
        try:
            return pickle.loads(data)
        except Exception:
            return json.loads(data.decode())
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Establece un valor en cache"""
        with self.lock:
            # Verificar tamaño máximo
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
            
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at
            )
            
            self.cache[key] = entry
            logger.debug(f"Cache set: {key}")
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del cache"""
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            entry = self.cache[key]
            
            # Verificar expiración
            if entry.is_expired():
                del self.cache[key]
                self.miss_count += 1
                return None
            
            # Actualizar estadísticas
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self.hit_count += 1
            
            return entry.value
    
    def delete(self, key: str) -> bool:
        """Elimina un valor del cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"Cache delete: {key}")
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Verifica si una clave existe"""
        with self.lock:
            if key not in self.cache:
                return False
            
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                return False
            
            return True
    
    def expire(self, key: str, ttl: int) -> bool:
        """Establece TTL para una clave"""
        with self.lock:
            if key not in self.cache:
                return False
            
            entry = self.cache[key]
            entry.expires_at = datetime.now() + timedelta(seconds=ttl)
            return True
    
    def ttl(self, key: str) -> Optional[int]:
        """Obtiene TTL restante de una clave"""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if entry.expires_at is None:
                return -1  # Sin expiración
            
            if entry.is_expired():
                del self.cache[key]
                return None
            
            remaining = (entry.expires_at - datetime.now()).total_seconds()
            return int(max(0, remaining))
    
    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Obtiene claves (con patrón opcional)"""
        with self.lock:
            keys = list(self.cache.keys())
            
            # Filtrar expiradas
            valid_keys = []
            for key in keys:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                else:
                    valid_keys.append(key)
            
            # Filtrar por patrón
            if pattern:
                import re
                pattern_regex = pattern.replace("*", ".*")
                valid_keys = [k for k in valid_keys if re.match(pattern_regex, k)]
            
            return valid_keys
    
    def clear(self):
        """Limpia todo el cache"""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def _evict_lru(self):
        """Elimina la entrada menos usada recientemente"""
        if not self.cache:
            return
        
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )
        del self.cache[lru_key]
        logger.debug(f"LRU eviction: {lru_key}")
    
    def cleanup_expired(self):
        """Limpia entradas expiradas"""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Incrementa un valor numérico"""
        with self.lock:
            current = self.get(key) or 0
            if isinstance(current, (int, float)):
                new_value = current + amount
                self.set(key, new_value)
                return new_value
            return 0
    
    def decrement(self, key: str, amount: int = 1) -> int:
        """Decrementa un valor numérico"""
        return self.increment(key, -amount)




