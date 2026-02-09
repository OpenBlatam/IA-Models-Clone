"""
Templates API Routes
====================

Endpoints para gestión de plantillas.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict
from pydantic import BaseModel

router = APIRouter(prefix="/templates", tags=["templates"])


class TemplateCreate(BaseModel):
    name: str
    content: str
    platform: Optional[str] = None
    variables: Optional[List[str]] = None
    category: Optional[str] = None


class TemplateRender(BaseModel):
    template_id: str
    variables: Dict[str, str]


def get_template_manager():
    """Dependency para obtener TemplateManager"""
    from ...services.template_manager import TemplateManager
    return TemplateManager()


@router.post("/", response_model=dict)
async def create_template(
    template: TemplateCreate,
    manager = Depends(get_template_manager)
):
    """Crear una nueva plantilla"""
    try:
        template_id = manager.create_template(
            name=template.name,
            content=template.content,
            platform=template.platform,
            variables=template.variables,
            category=template.category
        )
        return {"template_id": template_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[dict])
async def get_templates(
    query: Optional[str] = Query(None),
    platform: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    manager = Depends(get_template_manager)
):
    """Buscar plantillas"""
    try:
        templates = manager.search_templates(query, platform, category)
        return templates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{template_id}", response_model=dict)
async def get_template(
    template_id: str,
    manager = Depends(get_template_manager)
):
    """Obtener una plantilla específica"""
    try:
        template = manager.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Plantilla no encontrada")
        return template
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/render", response_model=dict)
async def render_template(
    render: TemplateRender,
    manager = Depends(get_template_manager)
):
    """Renderizar una plantilla con variables"""
    try:
        content = manager.render_template(render.template_id, render.variables)
        return {"rendered_content": content}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/{platform}", response_model=List[dict])
async def get_templates_by_platform(
    platform: str,
    manager = Depends(get_template_manager)
):
    """Obtener plantillas para una plataforma"""
    try:
        templates = manager.get_templates_by_platform(platform)
        return templates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{template_id}")
async def delete_template(
    template_id: str,
    manager = Depends(get_template_manager)
):
    """Eliminar una plantilla"""
    try:
        success = manager.delete_template(template_id)
        if not success:
            raise HTTPException(status_code=404, detail="Plantilla no encontrada")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




