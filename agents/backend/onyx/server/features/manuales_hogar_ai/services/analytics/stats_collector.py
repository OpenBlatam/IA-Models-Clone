"""
Stats Collector
==============

Recolector especializado de estadísticas.
"""

from typing import Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from ...core.base.service_base import BaseService
from ...database.models import Manual, ManualRating, ManualFavorite, UsageStats


class StatsCollector(BaseService):
    """Recolector de estadísticas."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar recolector.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def collect_manual_stats(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Recolectar estadísticas de manuales.
        
        Args:
            days: Días a considerar
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            total_manuals = await self._get_total_manuals()
            period_manuals = await self._get_period_manuals(start_date)
            total_views = await self._get_total_views()
            total_favorites = await self._get_total_favorites()
            total_shares = await self._get_total_shares()
            
            return {
                "total_manuals": total_manuals,
                "period_manuals": period_manuals,
                "total_views": total_views,
                "total_favorites": total_favorites,
                "total_shares": total_shares,
            }
        except Exception as e:
            self.log_error(f"Error recolectando stats: {str(e)}")
            return {}
    
    async def collect_rating_stats(self) -> Dict[str, Any]:
        """Recolectar estadísticas de ratings."""
        try:
            total_ratings = await self._get_total_ratings()
            global_avg_rating = await self._get_global_avg_rating()
            
            return {
                "total_ratings": total_ratings,
                "global_avg_rating": round(global_avg_rating, 2),
            }
        except Exception as e:
            self.log_error(f"Error recolectando rating stats: {str(e)}")
            return {}
    
    async def collect_category_stats(
        self,
        start_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """Recolectar estadísticas por categoría."""
        try:
            result = await self.db.execute(
                select(
                    Manual.category,
                    func.count(Manual.id).label('count'),
                    func.avg(Manual.average_rating).label('avg_rating'),
                    func.sum(Manual.view_count).label('total_views')
                ).where(
                    Manual.created_at >= start_date
                ).group_by(Manual.category)
            )
            
            category_stats = {}
            for row in result.all():
                category_stats[row.category] = {
                    "count": row.count,
                    "avg_rating": round(row.avg_rating or 0.0, 2),
                    "total_views": row.total_views or 0,
                }
            
            return category_stats
        except Exception as e:
            self.log_error(f"Error recolectando category stats: {str(e)}")
            return {}
    
    async def _get_total_manuals(self) -> int:
        """Obtener total de manuales."""
        result = await self.db.execute(select(func.count(Manual.id)))
        return result.scalar() or 0
    
    async def _get_period_manuals(self, start_date: datetime) -> int:
        """Obtener manuales del período."""
        result = await self.db.execute(
            select(func.count(Manual.id)).where(Manual.created_at >= start_date)
        )
        return result.scalar() or 0
    
    async def _get_total_views(self) -> int:
        """Obtener total de vistas."""
        result = await self.db.execute(select(func.sum(Manual.view_count)))
        return result.scalar() or 0
    
    async def _get_total_favorites(self) -> int:
        """Obtener total de favoritos."""
        result = await self.db.execute(select(func.sum(Manual.favorite_count)))
        return result.scalar() or 0
    
    async def _get_total_shares(self) -> int:
        """Obtener total de compartidos."""
        result = await self.db.execute(select(func.sum(Manual.share_count)))
        return result.scalar() or 0
    
    async def _get_total_ratings(self) -> int:
        """Obtener total de ratings."""
        result = await self.db.execute(select(func.count(ManualRating.id)))
        return result.scalar() or 0
    
    async def _get_global_avg_rating(self) -> float:
        """Obtener promedio global de rating."""
        result = await self.db.execute(
            select(func.avg(Manual.average_rating)).where(Manual.rating_count > 0)
        )
        return result.scalar() or 0.0

