"""
Similarity Engine
================

Motor especializado para encontrar manuales similares.
"""

from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from ...core.base.service_base import BaseService
from ...database.models import Manual


class SimilarityEngine(BaseService):
    """Motor para encontrar manuales similares."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar motor.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def find_similar(
        self,
        manual_id: int,
        limit: int = 5
    ) -> List[Manual]:
        """
        Encontrar manuales similares.
        
        Args:
            manual_id: ID del manual de referencia
            limit: Número de recomendaciones
        
        Returns:
            Lista de manuales similares
        """
        try:
            reference_manual = await self._get_reference_manual(manual_id)
            if not reference_manual:
                return []
            
            similar_manuals = await self._query_similar_manuals(
                reference_manual, manual_id, limit
            )
            
            self.log_info(f"Encontrados {len(similar_manuals)} manuales similares para {manual_id}")
            return similar_manuals
        
        except Exception as e:
            self.log_error(f"Error encontrando similares: {str(e)}")
            return []
    
    async def _get_reference_manual(self, manual_id: int) -> Manual:
        """Obtener manual de referencia."""
        result = await self.db.execute(
            select(Manual).where(Manual.id == manual_id)
        )
        return result.scalar_one_or_none()
    
    async def _query_similar_manuals(
        self,
        reference_manual: Manual,
        exclude_id: int,
        limit: int
    ) -> List[Manual]:
        """Consultar manuales similares."""
        result = await self.db.execute(
            select(Manual).where(
                and_(
                    Manual.id != exclude_id,
                    Manual.category == reference_manual.category,
                    Manual.is_public == True
                )
            ).order_by(
                desc(Manual.average_rating),
                desc(Manual.view_count)
            ).limit(limit)
        )
        return list(result.scalars().all())

