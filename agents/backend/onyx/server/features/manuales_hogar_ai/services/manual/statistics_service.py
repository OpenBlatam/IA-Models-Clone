"""
Statistics Service
=================

Servicio especializado para estadísticas.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from ...core.base.service_base import BaseService
from ...database.models import Manual, UsageStats


class StatisticsService(BaseService):
    """Servicio para estadísticas."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar servicio de estadísticas.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Obtener estadísticas de uso.
        
        Args:
            days: Número de días a considerar
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            total_manuals = await self._get_total_manuals()
            category_stats = await self._get_category_stats(start_date)
            total_tokens = await self._get_total_tokens()
            top_models = await self._get_top_models(start_date)
            
            return {
                "total_manuals": total_manuals,
                "category_stats": category_stats,
                "total_tokens": total_tokens,
                "top_models": top_models,
                "period_days": days
            }
        
        except Exception as e:
            self.log_error(f"Error obteniendo estadísticas: {str(e)}")
            return {}
    
    async def update_usage_stats(
        self,
        category: str,
        model_used: Optional[str],
        tokens_used: int,
        images_count: int
    ):
        """Actualizar estadísticas de uso."""
        try:
            today = datetime.now().date()
            
            stats_query = select(UsageStats).where(
                and_(
                    func.date(UsageStats.date) == today,
                    UsageStats.category == category,
                    UsageStats.model_used == model_used
                )
            )
            
            result = await self.db.execute(stats_query)
            stats = result.scalar_one_or_none()
            
            if stats:
                stats.total_requests += 1
                stats.total_tokens += tokens_used
                stats.total_images_processed += images_count
                stats.avg_tokens_per_request = stats.total_tokens / stats.total_requests
                stats.updated_at = datetime.now()
            else:
                stats = UsageStats(
                    date=datetime.now(),
                    category=category,
                    model_used=model_used,
                    total_requests=1,
                    total_tokens=tokens_used,
                    avg_tokens_per_request=tokens_used,
                    total_images_processed=images_count
                )
                self.db.add(stats)
            
            await self.db.commit()
        
        except Exception as e:
            self.log_warning(f"Error actualizando estadísticas: {str(e)}")
            await self.db.rollback()
    
    async def _get_total_manuals(self) -> int:
        """Obtener total de manuales."""
        result = await self.db.execute(select(func.count(Manual.id)))
        return result.scalar() or 0
    
    async def _get_category_stats(self, start_date: datetime) -> Dict[str, int]:
        """Obtener estadísticas por categoría."""
        category_query = select(
            Manual.category,
            func.count(Manual.id).label('count')
        ).where(
            Manual.created_at >= start_date
        ).group_by(Manual.category)
        
        result = await self.db.execute(category_query)
        return {
            row.category: row.count
            for row in result.all()
        }
    
    async def _get_total_tokens(self) -> int:
        """Obtener total de tokens."""
        result = await self.db.execute(select(func.sum(Manual.tokens_used)))
        return result.scalar() or 0
    
    async def _get_top_models(self, start_date: datetime) -> List[Dict[str, Any]]:
        """Obtener modelos más usados."""
        model_query = select(
            Manual.model_used,
            func.count(Manual.id).label('count')
        ).where(
            and_(
                Manual.created_at >= start_date,
                Manual.model_used.isnot(None)
            )
        ).group_by(Manual.model_used).order_by(desc('count')).limit(5)
        
        result = await self.db.execute(model_query)
        return [
            {"model": row.model_used, "count": row.count}
            for row in result.all()
        ]

