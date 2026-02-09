"""
Cache Service - Servicio de caché para respuestas de GitHub API.
"""

import json
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from cachetools import TTLCache
from config.logging_config import get_logger

logger = get_logger(__name__)


class CacheService:
    """Servicio de caché con TTL para respuestas de API."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 300  # 5 minutos por defecto
    ):
        """
        Inicializar servicio de caché con validaciones.

        Args:
            max_size: Tamaño máximo de la caché (debe ser > 0)
            default_ttl: TTL por defecto en segundos (debe ser > 0)

        Raises:
            ValueError: Si max_size o default_ttl son inválidos
        """
        # Validaciones
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError(f"max_size debe ser un entero positivo, recibido: {max_size}")
        
        if not isinstance(default_ttl, int) or default_ttl <= 0:
            raise ValueError(f"default_ttl debe ser un entero positivo, recibido: {default_ttl}")
        
        self.default_ttl = default_ttl
        self.cache: TTLCache[str, Dict[str, Any]] = TTLCache(
            maxsize=max_size,
            ttl=default_ttl
        )
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0
        }
        
        logger.info(
            f"CacheService inicializado: max_size={max_size}, default_ttl={default_ttl}s"
        )

    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor de la caché con validación.

        Args:
            key: Clave del valor a obtener (debe ser string no vacío)

        Returns:
            Valor almacenado o None si no existe o expiró

        Raises:
            ValueError: Si key es inválido
        """
        # Validación
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError(f"Key debe ser un string no vacío, recibido: {key}")
        
        key = key.strip()
        
        try:
            value = self.cache.get(key)
            if value is not None:
                self.stats["hits"] += 1
                logger.debug(f"✅ Cache hit para clave: {key}")
                return value.get("data") if isinstance(value, dict) else value
            else:
                self.stats["misses"] += 1
                logger.debug(f"❌ Cache miss para clave: {key}")
                return None
        except KeyError:
            self.stats["misses"] += 1
            logger.debug(f"❌ Cache miss (KeyError) para clave: {key}")
            return None
        except Exception as e:
            logger.warning(f"Error inesperado al obtener de caché (key={key}): {e}", exc_info=True)
            self.stats["misses"] += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Almacenar valor en la caché con validación.

        Args:
            key: Clave del valor (debe ser string no vacío)
            value: Valor a almacenar
            ttl: TTL en segundos (opcional, debe ser > 0 si se proporciona, usa default_ttl si no)

        Raises:
            ValueError: Si key o ttl son inválidos
        """
        # Validación de key
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError(f"Key debe ser un string no vacío, recibido: {key}")
        
        key = key.strip()
        
        # Validación de ttl
        if ttl is not None:
            if not isinstance(ttl, int) or ttl <= 0:
                raise ValueError(f"TTL debe ser un entero positivo, recibido: {ttl}")
        
        try:
            cache_value = {
                "data": value,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl or self.default_ttl
            }
            self.cache[key] = cache_value
            self.stats["sets"] += 1
            logger.debug(f"✅ Valor almacenado en caché: {key} (ttl={cache_value['ttl']}s)")
        except Exception as e:
            logger.error(f"Error al almacenar en caché (key={key}): {e}", exc_info=True)
            raise

    def delete(self, key: str) -> bool:
        """
        Eliminar valor de la caché con validación.

        Args:
            key: Clave a eliminar (debe ser string no vacío)

        Returns:
            True si se eliminó, False si no existía

        Raises:
            ValueError: Si key es inválido
        """
        # Validación
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError(f"Key debe ser un string no vacío, recibido: {key}")
        
        key = key.strip()
        
        try:
            del self.cache[key]
            logger.debug(f"✅ Valor eliminado de caché: {key}")
            return True
        except KeyError:
            logger.debug(f"⚠️  Intento de eliminar clave inexistente: {key}")
            return False
        except Exception as e:
            logger.warning(f"Error inesperado al eliminar de caché (key={key}): {e}", exc_info=True)
            return False

    def clear(self) -> None:
        """Limpiar toda la caché."""
        self.cache.clear()
        self.stats["evictions"] += 1
        logger.info("Caché limpiada completamente")

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de la caché.

        Returns:
            Diccionario con estadísticas
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            (self.stats["hits"] / total_requests * 100)
            if total_requests > 0
            else 0
        )

        return {
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "evictions": self.stats["evictions"],
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }

    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generar clave de caché consistente con validación.

        Args:
            prefix: Prefijo para la clave (debe ser string no vacío)
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre

        Returns:
            Clave generada

        Raises:
            ValueError: Si prefix es inválido
        """
        # Validación
        if not prefix or not isinstance(prefix, str) or not prefix.strip():
            raise ValueError(f"Prefix debe ser un string no vacío, recibido: {prefix}")
        
        prefix = prefix.strip()
        
        key_parts = [prefix]
        if args:
            key_parts.extend(str(arg) for arg in args)
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend(f"{k}={v}" for k, v in sorted_kwargs)
        
        generated_key = ":".join(key_parts)
        logger.debug(f"Clave generada: {generated_key}")
        return generated_key

