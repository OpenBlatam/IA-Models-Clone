"""
Popularity Engine
================

Motor especializado para encontrar manuales populares.
"""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from ...core.base.service_base import BaseService
from ...database.models import Manual


class PopularityEngine(BaseService):
    """Motor para encontrar manuales populares."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar motor.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def find_popular(
        self,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Manual]:
        """
        Encontrar manuales populares.
        
        Args:
            category: Filtrar por categoría (opcional)
            limit: Número de manuales
        
        Returns:
            Lista de manuales populares
        """
        try:
            query = self._build_popularity_query(category)
            query = query.order_by(
                desc(Manual.view_count),
                desc(Manual.average_rating),
                desc(Manual.favorite_count)
            ).limit(limit)
            
            result = await self.db.execute(query)
            manuals = list(result.scalars().all())
            
            self.log_info(f"Encontrados {len(manuals)} manuales populares")
            return manuals
        
        except Exception as e:
            self.log_error(f"Error encontrando populares: {str(e)}")
            return []
    
    def _build_popularity_query(self, category: Optional[str]):
        """Construir query de popularidad."""
        query = select(Manual).where(Manual.is_public == True)
        
        if category:
            query = query.where(Manual.category == category)
        
        return query

