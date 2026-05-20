"""
Recommendation Service
=====================

Servicio principal para recomendaciones de manuales.
"""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.base.service_base import BaseService
from ...database.models import Manual
from .similarity_engine import SimilarityEngine
from .popularity_engine import PopularityEngine


class RecommendationService(BaseService):
    """Servicio para generar recomendaciones."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar servicio.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.similarity_engine = SimilarityEngine(db)
        self.popularity_engine = PopularityEngine(db)
    
    async def get_similar_manuals(
        self,
        manual_id: int,
        limit: int = 5
    ) -> List[Manual]:
        """
        Obtener manuales similares.
        
        Args:
            manual_id: ID del manual de referencia
            limit: Número de recomendaciones
        
        Returns:
            Lista de manuales similares
        """
        return await self.similarity_engine.find_similar(manual_id, limit)
    
    async def get_popular_manuals(
        self,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Manual]:
        """
        Obtener manuales populares.
        
        Args:
            category: Filtrar por categoría (opcional)
            limit: Número de manuales
        
        Returns:
            Lista de manuales populares
        """
        return await self.popularity_engine.find_popular(category, limit)
    
    async def get_trending_manuals(
        self,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Manual]:
        """
        Obtener manuales en tendencia.
        
        Args:
            category: Filtrar por categoría (opcional)
            limit: Número de manuales
        
        Returns:
            Lista de manuales en tendencia
        """
        try:
            from sqlalchemy import select, desc, func
            from datetime import datetime, timedelta
            from ...database.models import Manual
            
            # Manuales con más vistas en los últimos 7 días
            seven_days_ago = datetime.now() - timedelta(days=7)
            
            query = select(Manual).where(
                Manual.is_public == True,
                Manual.created_at >= seven_days_ago
            )
            
            if category:
                query = query.where(Manual.category == category)
            
            query = query.order_by(
                desc(Manual.view_count),
                desc(Manual.favorite_count)
            ).limit(limit)
            
            result = await self.db.execute(query)
            manuals = list(result.scalars().all())
            
            self.log_info(f"Encontrados {len(manuals)} manuales en tendencia")
            return manuals
        
        except Exception as e:
            self.log_error(f"Error encontrando tendencias: {str(e)}")
            return []

