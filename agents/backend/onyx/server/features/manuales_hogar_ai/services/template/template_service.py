"""
Template Service
===============

Servicio principal para gestión de plantillas.
"""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.base.service_base import BaseService
from ...database.models import ManualTemplate
from .template_repository import TemplateRepository


class TemplateService(BaseService):
    """Servicio para gestionar plantillas."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar servicio.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.repository = TemplateRepository(db)
    
    async def create_template(
        self,
        name: str,
        category: str,
        template_content: str,
        description: Optional[str] = None,
        is_public: bool = True
    ) -> ManualTemplate:
        """
        Crear plantilla.
        
        Args:
            name: Nombre de la plantilla
            category: Categoría
            template_content: Contenido de la plantilla
            description: Descripción (opcional)
            is_public: Si es pública
        
        Returns:
            Plantilla creada
        """
        try:
            template = ManualTemplate(
                name=name,
                category=category,
                template_content=template_content,
                description=description,
                is_public=is_public
            )
            
            template = await self.repository.save(template)
            
            self.log_info(f"Plantilla creada: {name}, Categoría: {category}")
            return template
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error creando plantilla: {str(e)}")
            raise
    
    async def get_template(
        self,
        template_id: int
    ) -> Optional[ManualTemplate]:
        """
        Obtener plantilla por ID.
        
        Args:
            template_id: ID de la plantilla
        
        Returns:
            Plantilla o None
        """
        return await self.repository.get_by_id(template_id)
    
    async def get_templates_by_category(
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
        return await self.repository.get_by_category(category, limit)
    
    async def get_all_templates(
        self,
        limit: int = 50
    ) -> List[ManualTemplate]:
        """
        Obtener todas las plantillas públicas.
        
        Args:
            limit: Límite de resultados
        
        Returns:
            Lista de plantillas
        """
        return await self.repository.get_all_public(limit)

