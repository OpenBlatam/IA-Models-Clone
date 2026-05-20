"""
Rutas para Plantillas de Análisis
==================================

Endpoints para gestionar plantillas de análisis personalizadas.
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.analysis_templates import get_template_manager, TemplateManager
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/templates",
    tags=["Analysis Templates"]
)


class CreateTemplateRequest(BaseModel):
    """Request para crear plantilla"""
    name: str = Field(..., description="Nombre de la plantilla")
    description: str = Field(..., description="Descripción")
    tasks: List[str] = Field(..., description="Lista de tareas")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parámetros personalizados")


class ApplyTemplateRequest(BaseModel):
    """Request para aplicar plantilla"""
    template_name: str = Field(..., description="Nombre de la plantilla")
    content: str = Field(..., description="Contenido del documento")


@router.get("/")
async def list_templates(
    manager: TemplateManager = Depends(get_template_manager)
):
    """Listar todas las plantillas"""
    return {"templates": manager.list_templates()}


@router.get("/{template_name}")
async def get_template(
    template_name: str,
    manager: TemplateManager = Depends(get_template_manager)
):
    """Obtener plantilla específica"""
    template = manager.get_template(template_name)
    if not template:
        raise HTTPException(status_code=404, detail=f"Plantilla no encontrada: {template_name}")
    
    return {
        "name": template.name,
        "description": template.description,
        "tasks": template.tasks,
        "parameters": template.parameters,
        "enabled": template.enabled,
        "created_at": template.created_at,
        "updated_at": template.updated_at
    }


@router.post("/")
async def create_template(
    request: CreateTemplateRequest,
    manager: TemplateManager = Depends(get_template_manager)
):
    """Crear nueva plantilla"""
    try:
        template = manager.create_template(
            request.name,
            request.description,
            request.tasks,
            request.parameters
        )
        
        return {
            "status": "created",
            "template": {
                "name": template.name,
                "description": template.description,
                "tasks": template.tasks
            }
        }
    except Exception as e:
        logger.error(f"Error creando plantilla: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply")
async def apply_template(
    request: ApplyTemplateRequest,
    manager: TemplateManager = Depends(get_template_manager),
    analyzer = Depends(get_analyzer)
):
    """Aplicar plantilla a un documento"""
    try:
        result = await manager.apply_template(
            request.template_name,
            request.content,
            analyzer
        )
        
        return result
    except Exception as e:
        logger.error(f"Error aplicando plantilla: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{template_name}")
async def delete_template(
    template_name: str,
    manager: TemplateManager = Depends(get_template_manager)
):
    """Eliminar plantilla"""
    try:
        manager.delete_template(template_name)
        return {"status": "deleted", "template": template_name}
    except Exception as e:
        logger.error(f"Error eliminando plantilla: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















