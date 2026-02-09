"""
Sistema de caché para perfiles y contenido
Optimizado para velocidad con caché en memoria y orjson
"""

import logging
import hashlib
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    import json
    ORJSON_AVAILABLE = False

from ..config import get_settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Gestor de caché para perfiles y contenido
    Optimizado con caché en memoria (LRU) + disco
    """
    
    def __init__(self, memory_cache_size: int = 1000):
        """
        Inicializa el gestor de caché
        
        Args:
            memory_cache_size: Tamaño del caché en memoria (default: 1000)
        """
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.storage_path) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = timedelta(hours=24)  # 24 horas por defecto
        self._memory_cache: Dict[str, tuple[Any, datetime]] = {}
        self._memory_cache_size = memory_cache_size
    
    def _get_cache_key(self, platform: str, identifier: str) -> str:
        """Genera clave de caché"""
        key_string = f"{platform}:{identifier}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Obtiene ruta del archivo de caché"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, platform: str, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene valor del caché (optimizado: memoria primero, luego disco)
        
        Args:
            platform: Plataforma (tiktok, instagram, youtube)
            identifier: Identificador (username, channel_id)
            
        Returns:
            Datos cacheados o None si no existe/expirado
        """
        cache_key = self._get_cache_key(platform, identifier)
        
        # Verificar caché en memoria primero (más rápido)
        if cache_key in self._memory_cache:
            data, expiry = self._memory_cache[cache_key]
            if datetime.now() < expiry:
                return data
            else:
                # Expirado, eliminar de memoria
                del self._memory_cache[cache_key]
        
        # Verificar caché en disco
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            # Leer con orjson si está disponible (más rápido)
            if ORJSON_AVAILABLE:
                with open(cache_path, "rb") as f:
                    data = orjson.loads(f.read())
            else:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            
            # Verificar expiración
            cached_at = datetime.fromisoformat(data.get("cached_at"))
            ttl = timedelta(seconds=data.get("ttl", self.default_ttl.total_seconds()))
            
            if datetime.now() - cached_at > ttl:
                logger.info(f"Cache expired for {platform}:{identifier}")
                cache_path.unlink()  # Eliminar archivo expirado
                return None
            
            value = data.get("value")
            
            # Guardar en caché de memoria para acceso rápido
            expiry = datetime.now() + ttl
            if len(self._memory_cache) >= self._memory_cache_size:
                # Eliminar el más antiguo
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]
            self._memory_cache[cache_key] = (value, expiry)
            
            logger.debug(f"Cache hit for {platform}:{identifier}")
            return value
            
        except Exception as e:
            logger.error(f"Error reading cache for {platform}:{identifier}: {e}")
            return None
    
    def set(
        self, 
        platform: str, 
        identifier: str, 
        value: Dict[str, Any],
        ttl: Optional[timedelta] = None
    ):
        """
        Guarda valor en caché (memoria + disco)
        
        Args:
            platform: Plataforma
            identifier: Identificador
            value: Valor a cachear
            ttl: Time to live (opcional)
        """
        cache_key = self._get_cache_key(platform, identifier)
        cache_path = self._get_cache_path(cache_key)
        
        ttl = ttl or self.default_ttl
        expiry = datetime.now() + ttl
        
        # Guardar en caché de memoria primero (más rápido)
        if len(self._memory_cache) >= self._memory_cache_size:
            # Eliminar el más antiguo
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        self._memory_cache[cache_key] = (value, expiry)
        
        # Guardar en disco (persistencia)
        try:
            data = {
                "platform": platform,
                "identifier": identifier,
                "value": value,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl.total_seconds()
            }
            
            # Usar orjson si está disponible (2-3x más rápido)
            if ORJSON_AVAILABLE:
                with open(cache_path, "wb") as f:
                    f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            else:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Cached {platform}:{identifier}")
            
        except Exception as e:
            logger.error(f"Error caching {platform}:{identifier}: {e}")
    
    def delete(self, platform: str, identifier: str):
        """Elimina entrada del caché (memoria + disco)"""
        cache_key = self._get_cache_key(platform, identifier)
        
        # Eliminar de memoria
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        # Eliminar de disco
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Deleted cache for {platform}:{identifier}")
    
    def clear(self, platform: Optional[str] = None):
        """
        Limpia caché (memoria + disco)
        
        Args:
            platform: Si se especifica, solo limpia esa plataforma
        """
        # Limpiar memoria
        if platform:
            # Eliminar solo entradas de la plataforma
            keys_to_delete = [
                key for key in self._memory_cache.keys()
                if key.startswith(self._get_cache_key(platform, "").split(':')[0])
            ]
            for key in keys_to_delete:
                del self._memory_cache[key]
        else:
            self._memory_cache.clear()
        
        # Limpiar disco
        if platform:
            pattern = f"*_{platform}_*.json"
        else:
            pattern = "*.json"
        
        deleted = 0
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            deleted += 1
        
        logger.info(f"Cleared {deleted} cache entries" + (f" for {platform}" if platform else ""))




