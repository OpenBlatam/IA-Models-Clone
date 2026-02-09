"""
Analytics Service
================

Servicio principal para analytics y reportes.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.base.service_base import BaseService
from .stats_collector import StatsCollector
from .report_generator import ReportGenerator


class AnalyticsService(BaseService):
    """Servicio para analytics avanzado."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar servicio.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.stats_collector = StatsCollector(db)
        self.report_generator = ReportGenerator(db)
    
    async def get_comprehensive_stats(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Obtener estadísticas comprehensivas.
        
        Args:
            days: Días a considerar
        
        Returns:
            Diccionario con estadísticas completas
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            manual_stats = await self.stats_collector.collect_manual_stats(days)
            rating_stats = await self.stats_collector.collect_rating_stats()
            category_stats = await self.stats_collector.collect_category_stats(start_date)
            
            return {
                **manual_stats,
                **rating_stats,
                "category_stats": category_stats,
                "period_days": days,
            }
        except Exception as e:
            self.log_error(f"Error obteniendo stats comprehensivas: {str(e)}")
            return {}
    
    async def get_trending_report(
        self,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Obtener reporte de tendencias."""
        return await self.report_generator.generate_trending_report(days)
    
    async def get_top_rated_report(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Obtener reporte de mejor calificados."""
        return await self.report_generator.generate_top_rated_report(limit)
    
    async def get_category_performance_report(
        self,
        days: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """Obtener reporte de rendimiento por categoría."""
        return await self.report_generator.generate_category_performance_report(days)

