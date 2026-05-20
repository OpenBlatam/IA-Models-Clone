"""
Cache Manager - Gestor de Cache
================================

Gestiona cache de proyectos generados para mejorar rendimiento.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
Refactored with improved error handling and file operations.
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

from .file_operations import read_json, write_json, FileOperationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheEntry:
    """
    Entrada de cache.
    Inmutable para mejor seguridad.
    """
    cache_key: str
    description: str
    config: Dict[str, Any]
    project_info: Dict[str, Any]
    created_at: str


def _generate_cache_key(description: str, config: Dict[str, Any]) -> str:
    """
    Genera una clave de cache única (función pura).
    
    Args:
        description: Descripción del proyecto
        config: Configuración del proyecto
        
    Returns:
        Clave de cache MD5
    """
    if not description:
        raise ValueError("description cannot be empty")
    
    cache_data = {
        "description": description,
        "config": config,
    }
    cache_string = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_string.encode()).hexdigest()


def _is_cache_expired(created_at: str, expiration_days: int = 7) -> bool:
    """
    Verifica si una entrada de cache está expirada (función pura).
    
    Args:
        created_at: Fecha de creación en formato ISO
        expiration_days: Días de expiración
        
    Returns:
        True si está expirada, False en caso contrario
    """
    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        return (now - created) > timedelta(days=expiration_days)
    except (ValueError, TypeError):
        return True


def _read_cache_file(cache_file: Path) -> Optional[Dict[str, Any]]:
    """
    Lee un archivo de cache (función pura).
    
    Args:
        cache_file: Archivo de cache
        
    Returns:
        Datos del cache o None si hay error
    """
    if not cache_file.exists():
        return None
    
    try:
        return read_json(cache_file, default=None)
    except FileOperationError as e:
        logger.warning(f"Error reading cache file {cache_file}: {e}")
        return None


def _write_cache_file(cache_file: Path, cache_data: Dict[str, Any]) -> bool:
    """
    Escribe un archivo de cache (función pura).
    
    Args:
        cache_file: Archivo de cache
        cache_data: Datos a escribir
        
    Returns:
        True si se escribió exitosamente, False en caso contrario
    """
    try:
        write_json(cache_file, cache_data)
        return True
    except FileOperationError as e:
        logger.warning(f"Error writing cache file {cache_file}: {e}")
        return False


class CacheManager:
    """
    Gestor de cache para proyectos.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """
        Inicializa el gestor de cache.

        Args:
            cache_dir: Directorio para almacenar cache
            
        Raises:
            OSError: Si no se puede crear el directorio
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache"
        
        self.cache_dir = Path(cache_dir)
        
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise
    
    async def get_cached_project(
        self,
        description: str,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene un proyecto del cache si existe.

        Args:
            description: Descripción del proyecto
            config: Configuración del proyecto

        Returns:
            Información del proyecto cacheado o None
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not description:
            raise ValueError("description cannot be empty")
        
        if not config:
            raise ValueError("config cannot be empty")
        
        try:
            cache_key = _generate_cache_key(description, config)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            cache_data = _read_cache_file(cache_file)
            if cache_data is None:
                return None
            
            # Verificar expiración
            created_at = cache_data.get("created_at", "")
            if _is_cache_expired(created_at, expiration_days=7):
                cache_file.unlink()
                return None
            
            logger.info(f"Project found in cache: {cache_key}")
            return cache_data.get("project_info")
        
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            return None
    
    async def cache_project(
        self,
        description: str,
        config: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> bool:
        """
        Guarda un proyecto en cache.

        Args:
            description: Descripción del proyecto
            config: Configuración del proyecto
            project_info: Información del proyecto generado
            
        Returns:
            True si se guardó exitosamente, False en caso contrario
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not description:
            raise ValueError("description cannot be empty")
        
        if not config:
            raise ValueError("config cannot be empty")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        try:
            cache_key = _generate_cache_key(description, config)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            cache_data = {
                "cache_key": cache_key,
                "description": description,
                "config": config,
                "project_info": project_info,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            
            success = _write_cache_file(cache_file, cache_data)
            if success:
                logger.info(f"Project saved in cache: {cache_key}")
            
            return success
        
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
            return False
    
    async def clear_cache(self, older_than_days: int = 7) -> Dict[str, int]:
        """
        Limpia el cache eliminando entradas antiguas.

        Args:
            older_than_days: Días de antigüedad para eliminar
            
        Returns:
            Diccionario con estadísticas de limpieza
            
        Raises:
            ValueError: Si older_than_days es inválido
        """
        if older_than_days < 0:
            raise ValueError("older_than_days must be non-negative")
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
            cleared = 0
            
            for cache_file in self.cache_dir.glob("*.json"):
                cache_data = _read_cache_file(cache_file)
                if cache_data is None:
                    continue
                
                created_at = cache_data.get("created_at", "")
                try:
                    created = datetime.fromisoformat(created_at)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    
                    if created < cutoff_date:
                        cache_file.unlink()
                        cleared += 1
                except (ValueError, TypeError):
                    # Si no se puede parsear la fecha, eliminar el archivo
                    cache_file.unlink()
                    cleared += 1
            
            logger.info(f"Cache cleared: {cleared} entries removed")
            return {"cleared": cleared}
        
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return {"cleared": 0}
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache.
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "total_entries": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_dir": str(self.cache_dir),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "total_entries": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "cache_dir": str(self.cache_dir),
            }
