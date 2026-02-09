"""
Cache Manager - Sistema de cache para mejoras y embeddings
==========================================================
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Gestiona cache de mejoras de código y embeddings.
    """
    
    def __init__(self, cache_dir: str = "data/cache", ttl_hours: int = 24):
        """
        Inicializar gestor de cache.
        
        Args:
            cache_dir: Directorio de cache
            ttl_hours: Tiempo de vida del cache en horas
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        
        # Cache en memoria
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
    
    def _generate_key(self, content: str, context: Optional[str] = None) -> str:
        """Genera clave de cache desde contenido"""
        key_string = content
        if context:
            key_string += context
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_improvement(self, code: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Obtiene mejora desde cache si existe.
        
        Args:
            code: Código original
            context: Contexto adicional
            
        Returns:
            Mejora cacheada o None
        """
        try:
            cache_key = self._generate_key(code, context)
            
            # Verificar cache en memoria
            if cache_key in self.memory_cache:
                cached = self.memory_cache[cache_key]
                if self._is_valid(cached):
                    logger.info(f"Cache hit (memory): {cache_key[:8]}")
                    return cached.get("data")
                else:
                    del self.memory_cache[cache_key]
            
            # Verificar cache en disco
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                
                if self._is_valid(cached):
                    logger.info(f"Cache hit (disk): {cache_key[:8]}")
                    # Cargar a memoria
                    self.memory_cache[cache_key] = cached
                    return cached.get("data")
                else:
                    # Eliminar cache expirado
                    cache_file.unlink()
            
            return None
            
        except Exception as e:
            logger.warning(f"Error obteniendo cache: {e}")
            return None
    
    def cache_improvement(
        self,
        code: str,
        improvement: Dict[str, Any],
        context: Optional[str] = None
    ) -> bool:
        """
        Guarda mejora en cache.
        
        Args:
            code: Código original
            improvement: Mejora generada
            context: Contexto adicional
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            cache_key = self._generate_key(code, context)
            
            cached_data = {
                "key": cache_key,
                "data": improvement,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + self.ttl).isoformat()
            }
            
            # Guardar en memoria
            self.memory_cache[cache_key] = cached_data
            
            # Guardar en disco
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Mejora cacheada: {cache_key[:8]}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando cache: {e}")
            return False
    
    def _is_valid(self, cached: Dict[str, Any]) -> bool:
        """Verifica si el cache es válido (no expirado)"""
        try:
            expires_at_str = cached.get("expires_at")
            if not expires_at_str:
                return False
            
            expires_at = datetime.fromisoformat(expires_at_str)
            return datetime.now() < expires_at
            
        except Exception:
            return False
    
    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Limpia cache expirado o todo el cache.
        
        Args:
            older_than_hours: Limpiar cache más antiguo que X horas (None = todo)
            
        Returns:
            Número de archivos eliminados
        """
        try:
            deleted = 0
            
            # Limpiar memoria
            if older_than_hours is None:
                self.memory_cache.clear()
            else:
                cutoff = datetime.now() - timedelta(hours=older_than_hours)
                keys_to_delete = []
                for key, cached in self.memory_cache.items():
                    created_at = datetime.fromisoformat(cached.get("created_at", ""))
                    if created_at < cutoff:
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    del self.memory_cache[key]
            
            # Limpiar disco
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    
                    should_delete = False
                    if older_than_hours is None:
                        should_delete = True
                    else:
                        created_at = datetime.fromisoformat(cached.get("created_at", ""))
                        cutoff = datetime.now() - timedelta(hours=older_than_hours)
                        should_delete = created_at < cutoff
                    
                    if should_delete:
                        cache_file.unlink()
                        deleted += 1
                        
                except Exception as e:
                    logger.warning(f"Error procesando {cache_file}: {e}")
                    continue
            
            logger.info(f"Cache limpiado: {deleted} archivos eliminados")
            return deleted
            
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        try:
            disk_files = list(self.cache_dir.glob("*.json"))
            memory_count = len(self.memory_cache)
            
            total_size = sum(f.stat().st_size for f in disk_files)
            
            return {
                "memory_entries": memory_count,
                "disk_entries": len(disk_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_dir": str(self.cache_dir),
                "ttl_hours": self.ttl.total_seconds() / 3600
            }
        except Exception as e:
            logger.error(f"Error obteniendo stats: {e}")
            return {}




