"""
Manual Repository
================

Repository para acceso a datos de manuales.
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ...core.base.service_base import BaseService
from ...database.models import Manual


class ManualRepository(BaseService):
    """Repository para acceso a datos de manuales."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar repository.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def save(self, manual: Manual) -> Manual:
        """
        Guardar manual en base de datos.
        
        Args:
            manual: Instancia de Manual
        
        Returns:
            Manual guardado
        """
        try:
            self.db.add(manual)
            await self.db.commit()
            await self.db.refresh(manual)
            self.log_info(f"Manual guardado: ID {manual.id}")
            return manual
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error guardando manual: {str(e)}")
            raise
    
    async def get_by_id(self, manual_id: int) -> Optional[Manual]:
        """
        Obtener manual por ID.
        
        Args:
            manual_id: ID del manual
        
        Returns:
            Manual o None
        """
        try:
            result = await self.db.execute(
                select(Manual).where(Manual.id == manual_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self.log_error(f"Error obteniendo manual {manual_id}: {str(e)}")
            return None
    
    async def get_by_category(
        self,
        category: str,
        limit: int = 20
    ) -> list[Manual]:
        """
        Obtener manuales por categoría.
        
        Args:
            category: Categoría
            limit: Límite de resultados
        
        Returns:
            Lista de manuales
        """
        try:
            result = await self.db.execute(
                select(Manual)
                .where(Manual.category == category.lower())
                .where(Manual.is_public == True)
                .limit(limit)
            )
            return list(result.scalars().all())
        except Exception as e:
            self.log_error(f"Error obteniendo manuales por categoría: {str(e)}")
            return []

