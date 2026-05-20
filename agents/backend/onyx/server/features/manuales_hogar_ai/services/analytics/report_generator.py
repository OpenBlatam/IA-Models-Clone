"""
Report Generator
===============

Generador especializado de reportes.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from ...core.base.service_base import BaseService
from ...database.models import Manual, ManualRating


class ReportGenerator(BaseService):
    """Generador de reportes."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar generador.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def generate_trending_report(
        self,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Generar reporte de tendencias.
        
        Args:
            days: Días a considerar
        
        Returns:
            Lista de manuales en tendencia
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            result = await self.db.execute(
                select(Manual).where(
                    and_(
                        Manual.is_public == True,
                        Manual.created_at >= start_date
                    )
                ).order_by(
                    desc(Manual.view_count),
                    desc(Manual.favorite_count)
                ).limit(10)
            )
            
            manuals = list(result.scalars().all())
            
            return [
                {
                    "id": m.id,
                    "title": m.title,
                    "category": m.category,
                    "view_count": m.view_count,
                    "favorite_count": m.favorite_count,
                    "average_rating": m.average_rating,
                }
                for m in manuals
            ]
        except Exception as e:
            self.log_error(f"Error generando trending report: {str(e)}")
            return []
    
    async def generate_top_rated_report(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generar reporte de mejor calificados.
        
        Args:
            limit: Número de resultados
        
        Returns:
            Lista de manuales mejor calificados
        """
        try:
            result = await self.db.execute(
                select(Manual).where(
                    and_(
                        Manual.is_public == True,
                        Manual.rating_count >= 3
                    )
                ).order_by(
                    desc(Manual.average_rating),
                    desc(Manual.rating_count)
                ).limit(limit)
            )
            
            manuals = list(result.scalars().all())
            
            return [
                {
                    "id": m.id,
                    "title": m.title,
                    "category": m.category,
                    "average_rating": m.average_rating,
                    "rating_count": m.rating_count,
                }
                for m in manuals
            ]
        except Exception as e:
            self.log_error(f"Error generando top rated report: {str(e)}")
            return []
    
    async def generate_category_performance_report(
        self,
        days: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generar reporte de rendimiento por categoría.
        
        Args:
            days: Días a considerar
        
        Returns:
            Diccionario con rendimiento por categoría
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            result = await self.db.execute(
                select(
                    Manual.category,
                    func.count(Manual.id).label('count'),
                    func.avg(Manual.average_rating).label('avg_rating'),
                    func.sum(Manual.view_count).label('total_views'),
                    func.avg(Manual.view_count).label('avg_views')
                ).where(
                    Manual.created_at >= start_date
                ).group_by(Manual.category)
            )
            
            performance = {}
            for row in result.all():
                performance[row.category] = {
                    "count": row.count,
                    "avg_rating": round(row.avg_rating or 0.0, 2),
                    "total_views": row.total_views or 0,
                    "avg_views": round(row.avg_views or 0.0, 2),
                }
            
            return performance
        except Exception as e:
            self.log_error(f"Error generando category performance report: {str(e)}")
            return {}

