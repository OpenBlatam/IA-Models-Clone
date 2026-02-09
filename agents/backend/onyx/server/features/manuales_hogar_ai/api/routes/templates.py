"""
Rutas de Plantillas
===================

Endpoints para gestionar plantillas de manuales.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.session import get_async_session
from ...services.template_service import TemplateService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["templates"])


# Modelos Pydantic
class TemplateRequest(BaseModel):
    """Request para crear plantilla."""
    name: str = Field(..., description="Nombre de la plantilla")
    category: str = Field(..., description="Categoría")
    template_content: str = Field(..., description="Contenido de la plantilla")
    description: Optional[str] = Field(None, description="Descripción")
    is_public: bool = Field(True, description="Si es pública")


class TemplateResponse(BaseModel):
    """Response de plantilla."""
    id: int
    name: str
    category: str
    template_content: str
    description: Optional[str]
    is_public: bool
    usage_count: int
    created_at: str
    
    class Config:
        from_attributes = True


class ApplyTemplateRequest(BaseModel):
    """Request para aplicar plantilla."""
    variables: dict = Field(default_factory=dict, description="Variables para reemplazar")


# Dependencies
async def get_db_session() -> AsyncSession:
    """Obtener sesión de base de datos."""
    async for session in get_async_session():
        yield session


# Endpoints
@router.post("/templates", response_model=TemplateResponse)
async def create_template(
    request: TemplateRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Crear plantilla.
    
    - **name**: Nombre de la plantilla
    - **category**: Categoría
    - **template_content**: Contenido (puede usar {{variable}})
    - **description**: Descripción (opcional)
    - **is_public**: Si es pública
    """
    try:
        service = TemplateService(db)
        template = await service.create_template(
            name=request.name,
            category=request.category,
            template_content=request.template_content,
            description=request.description,
            is_public=request.is_public
        )
        
        return TemplateResponse(
            id=template.id,
            name=template.name,
            category=template.category,
            template_content=template.template_content,
            description=template.description,
            is_public=template.is_public,
            usage_count=template.usage_count,
            created_at=template.created_at.isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error creando plantilla: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creando plantilla: {str(e)}")


@router.get("/templates", response_model=List[TemplateResponse])
async def list_templates(
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Listar plantillas.
    
    - **category**: Filtrar por categoría (opcional)
    - **limit**: Límite de resultados
    """
    try:
        service = TemplateService(db)
        
        if category:
            templates = await service.get_templates_by_category(category, limit=limit)
        else:
            templates = await service.get_all_templates(limit=limit)
        
        return [
            TemplateResponse(
                id=t.id,
                name=t.name,
                category=t.category,
                template_content=t.template_content,
                description=t.description,
                is_public=t.is_public,
                usage_count=t.usage_count,
                created_at=t.created_at.isoformat()
            )
            for t in templates
        ]
    
    except Exception as e:
        logger.error(f"Error listando plantillas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listando plantillas: {str(e)}")


@router.get("/templates/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: int = Path(..., description="ID de la plantilla"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Obtener plantilla por ID.
    
    - **template_id**: ID de la plantilla
    """
    try:
        service = TemplateService(db)
        template = await service.get_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail="Plantilla no encontrada")
        
        return TemplateResponse(
            id=template.id,
            name=template.name,
            category=template.category,
            template_content=template.template_content,
            description=template.description,
            is_public=template.is_public,
            usage_count=template.usage_count,
            created_at=template.created_at.isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo plantilla: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo plantilla: {str(e)}")


@router.post("/templates/{template_id}/apply")
async def apply_template(
    template_id: int = Path(..., description="ID de la plantilla"),
    request: ApplyTemplateRequest = ...,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Aplicar plantilla con variables.
    
    - **template_id**: ID de la plantilla
    - **variables**: Diccionario de variables para reemplazar {{variable}}
    """
    try:
        service = TemplateService(db)
        template = await service.get_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail="Plantilla no encontrada")
        
        # Aplicar plantilla
        content = service.apply_template(template.template_content, request.variables)
        
        # Incrementar uso
        await service.increment_usage(template_id)
        
        return {
            "success": True,
            "template_id": template_id,
            "template_name": template.name,
            "content": content,
            "variables_used": list(request.variables.keys())
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error aplicando plantilla: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error aplicando plantilla: {str(e)}")




