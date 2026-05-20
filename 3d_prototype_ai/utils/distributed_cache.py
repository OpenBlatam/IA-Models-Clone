"""
Distributed Cache - Sistema de caché distribuido
=================================================
"""

import logging
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class DistributedCache:
    """Sistema de caché distribuido (preparado para Redis)"""
    
    def __init__(self, cache_dir: Optional[str] = None, use_redis: bool = False, redis_url: Optional[str] = None):
        self.use_redis = use_redis
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache en memoria como fallback
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Intentar conectar a Redis si está configurado
        self.redis_client = None
        if use_redis and redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Conectado a Redis")
            except Exception as e:
                logger.warning(f"No se pudo conectar a Redis: {e}. Usando caché en memoria.")
                self.use_redis = False
    
    def _get_key(self, key: str) -> str:
        """Genera una clave única"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Almacena un valor en el caché"""
        cache_key = self._get_key(key)
        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        
        cache_data = {
            "value": value,
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        if self.use_redis and self.redis_client:
            try:
                # Serializar valor
                serialized = json.dumps(value, default=str)
                self.redis_client.setex(cache_key, ttl_seconds, serialized)
                return
            except Exception as e:
                logger.warning(f"Error guardando en Redis: {e}. Usando fallback.")
        
        # Fallback: caché en memoria y disco
        self.memory_cache[cache_key] = cache_data
        
        # Guardar en disco también
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"Error guardando en disco: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del caché"""
        cache_key = self._get_key(key)
        
        if self.use_redis and self.redis_client:
            try:
                serialized = self.redis_client.get(cache_key)
                if serialized:
                    return json.loads(serialized)
            except Exception as e:
                logger.warning(f"Error leyendo de Redis: {e}. Usando fallback.")
        
        # Fallback: leer de memoria
        if cache_key in self.memory_cache:
            cache_data = self.memory_cache[cache_key]
            expires_at = datetime.fromisoformat(cache_data["expires_at"])
            
            if datetime.now() < expires_at:
                return cache_data["value"]
            else:
                del self.memory_cache[cache_key]
        
        # Fallback: leer de disco
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                expires_at = datetime.fromisoformat(cache_data["expires_at"])
                if datetime.now() < expires_at:
                    return cache_data["value"]
                else:
                    cache_file.unlink()  # Eliminar archivo expirado
            except Exception as e:
                logger.warning(f"Error leyendo de disco: {e}")
        
        return None
    
    def delete(self, key: str) -> bool:
        """Elimina un valor del caché"""
        cache_key = self._get_key(key)
        
        if self.use_redis and self.redis_client:
            try:
                return bool(self.redis_client.delete(cache_key))
            except Exception as e:
                logger.warning(f"Error eliminando de Redis: {e}")
        
        # Eliminar de memoria
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Eliminar de disco
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            cache_file.unlink()
            return True
        
        return False
    
    def clear(self, pattern: Optional[str] = None):
        """Limpia el caché"""
        if self.use_redis and self.redis_client:
            try:
                if pattern:
                    keys = self.redis_client.keys(f"*{pattern}*")
                    if keys:
                        self.redis_client.delete(*keys)
                else:
                    self.redis_client.flushdb()
                return
            except Exception as e:
                logger.warning(f"Error limpiando Redis: {e}")
        
        # Limpiar memoria
        if pattern:
            keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self.memory_cache[key]
        else:
            self.memory_cache.clear()
        
        # Limpiar disco
        if pattern:
            for cache_file in self.cache_dir.glob("*.json"):
                if pattern in cache_file.stem:
                    cache_file.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_files": len(list(self.cache_dir.glob("*.json"))),
            "using_redis": self.use_redis and self.redis_client is not None
        }
        
        if self.use_redis and self.redis_client:
            try:
                stats["redis_keys"] = self.redis_client.dbsize()
            except:
                pass
        
        return stats
    
    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Obtiene todas las claves (solo memoria/disco, no Redis)"""
        keys = list(self.memory_cache.keys())
        
        if pattern:
            keys = [k for k in keys if pattern in k]
        
        return keys




