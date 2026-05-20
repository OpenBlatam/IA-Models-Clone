"""
Cache Repository
================

Repository para acceso a datos de cache.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, delete, func

from ...core.base.service_base import BaseService
from ...database.models import ManualCache


class CacheRepository(BaseService):
    """Repository para acceso a datos de cache."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar repository.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def get_by_key(
        self,
        cache_key: str
    ) -> Optional[ManualCache]:
        """
        Obtener entrada de cache por clave.
        
        Args:
            cache_key: Clave de cache
        
        Returns:
            Entrada de cache o None
        """
        try:
            result = await self.db.execute(
                select(ManualCache).where(
                    and_(
                        ManualCache.cache_key == cache_key,
                        ManualCache.expires_at > datetime.now()
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self.log_error(f"Error obteniendo cache: {str(e)}")
            return None
    
    async def save(self, cache_entry: ManualCache) -> ManualCache:
        """
        Guardar entrada de cache.
        
        Args:
            cache_entry: Entrada de cache
        
        Returns:
            Entrada guardada
        """
        try:
            self.db.add(cache_entry)
            await self.db.commit()
            await self.db.refresh(cache_entry)
            return cache_entry
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error guardando cache: {str(e)}")
            raise
    
    async def update_hit(
        self,
        cache_entry: ManualCache
    ):
        """
        Actualizar contador de hits.
        
        Args:
            cache_entry: Entrada de cache
        """
        try:
            cache_entry.hit_count += 1
            cache_entry.last_accessed = datetime.now()
            await self.db.commit()
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error actualizando hit: {str(e)}")
    
    async def delete_expired(self) -> int:
        """
        Eliminar entradas expiradas.
        
        Returns:
            Número de entradas eliminadas
        """
        try:
            query = delete(ManualCache).where(
                ManualCache.expires_at < datetime.now()
            )
            result = await self.db.execute(query)
            await self.db.commit()
            return result.rowcount
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error eliminando expiradas: {str(e)}")
            return 0
    
    async def delete_all(self) -> int:
        """
        Eliminar todas las entradas.
        
        Returns:
            Número de entradas eliminadas
        """
        try:
            query = delete(ManualCache)
            result = await self.db.execute(query)
            await self.db.commit()
            return result.rowcount
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error eliminando todas: {str(e)}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del cache.
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            total = await self._get_total_count()
            expired = await self._get_expired_count()
            total_hits = await self._get_total_hits()
            category_stats = await self._get_category_stats()
            
            return {
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
                "total_hits": total_hits,
                "category_stats": category_stats
            }
        except Exception as e:
            self.log_error(f"Error obteniendo stats: {str(e)}")
            return {}
    
    async def _get_total_count(self) -> int:
        """Obtener total de entradas."""
        result = await self.db.execute(select(func.count(ManualCache.id)))
        return result.scalar() or 0
    
    async def _get_expired_count(self) -> int:
        """Obtener entradas expiradas."""
        result = await self.db.execute(
            select(func.count(ManualCache.id)).where(
                ManualCache.expires_at < datetime.now()
            )
        )
        return result.scalar() or 0
    
    async def _get_total_hits(self) -> int:
        """Obtener total de hits."""
        result = await self.db.execute(select(func.sum(ManualCache.hit_count)))
        return result.scalar() or 0
    
    async def _get_category_stats(self) -> Dict[str, int]:
        """Obtener estadísticas por categoría."""
        result = await self.db.execute(
            select(
                ManualCache.category,
                func.count(ManualCache.id).label('count')
            ).group_by(ManualCache.category)
        )
        return {
            row.category: row.count
            for row in result.all()
        }

