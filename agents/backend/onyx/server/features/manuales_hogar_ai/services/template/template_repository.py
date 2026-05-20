"""
Template Repository
==================

Repository para acceso a datos de plantillas.
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from ...core.base.service_base import BaseService
from ...database.models import ManualTemplate


class TemplateRepository(BaseService):
    """Repository para acceso a datos de plantillas."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar repository.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def save(self, template: ManualTemplate) -> ManualTemplate:
        """
        Guardar plantilla en base de datos.
        
        Args:
            template: Instancia de ManualTemplate
        
        Returns:
            Plantilla guardada
        """
        try:
            self.db.add(template)
            await self.db.commit()
            await self.db.refresh(template)
            self.log_info(f"Plantilla guardada: ID {template.id}")
            return template
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error guardando plantilla: {str(e)}")
            raise
    
    async def get_by_id(self, template_id: int) -> Optional[ManualTemplate]:
        """
        Obtener plantilla por ID.
        
        Args:
            template_id: ID de la plantilla
        
        Returns:
            Plantilla o None
        """
        try:
            result = await self.db.execute(
                select(ManualTemplate).where(ManualTemplate.id == template_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self.log_error(f"Error obteniendo plantilla: {str(e)}")
            return None
    
    async def get_by_category(
        self,
        category: str,
        limit: int = 20
    ) -> List[ManualTemplate]:
        """
        Obtener plantillas por categoría.
        
        Args:
            category: Categoría
            limit: Límite de resultados
        
        Returns:
            Lista de plantillas
        """
        try:
            result = await self.db.execute(
                select(ManualTemplate).where(
                    and_(
                        ManualTemplate.category == category,
                        ManualTemplate.is_public == True
                    )
                ).order_by(desc(ManualTemplate.created_at)).limit(limit)
            )
            return list(result.scalars().all())
        except Exception as e:
            self.log_error(f"Error obteniendo plantillas: {str(e)}")
            return []
    
    async def get_all_public(self, limit: int = 50) -> List[ManualTemplate]:
        """
        Obtener todas las plantillas públicas.
        
        Args:
            limit: Límite de resultados
        
        Returns:
            Lista de plantillas
        """
        try:
            result = await self.db.execute(
                select(ManualTemplate)
                .where(ManualTemplate.is_public == True)
                .order_by(desc(ManualTemplate.created_at))
                .limit(limit)
            )
            return list(result.scalars().all())
        except Exception as e:
            self.log_error(f"Error obteniendo plantillas públicas: {str(e)}")
            return []

