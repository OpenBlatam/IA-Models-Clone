"""
Templates endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.templates import TemplatesService

router = APIRouter()
templates_service = TemplatesService()


@router.get("/")
async def get_templates(template_type: Optional[str] = None) -> Dict[str, Any]:
    """Obtener plantillas"""
    try:
        if template_type:
            templates = templates_service.get_templates_by_type(template_type)
        else:
            templates = list(templates_service.templates.values())
        
        return {
            "templates": [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "type": t.template_type,
                    "variables": t.variables,
                }
                for t in templates
            ],
            "total": len(templates),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{template_id}")
async def get_template(template_id: str) -> Dict[str, Any]:
    """Obtener plantilla específica"""
    try:
        template = templates_service.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "type": template.template_type,
            "content": template.content,
            "variables": template.variables,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/render/{template_id}")
async def render_template(
    template_id: str,
    variables: Dict[str, str]
) -> Dict[str, Any]:
    """Renderizar plantilla con variables"""
    try:
        rendered = templates_service.render_template(template_id, variables)
        return {
            "template_id": template_id,
            "rendered_content": rendered,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




