"""
Gradio Integration endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.gradio_integration import GradioIntegrationService

router = APIRouter()
gradio_service = GradioIntegrationService()


@router.post("/create-interface")
async def create_interface(
    title: str,
    description: str,
    inputs: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    theme: str = "default"
) -> Dict[str, Any]:
    """Crear interfaz de Gradio"""
    try:
        interface = gradio_service.create_interface(
            title, description, inputs, outputs, None, None, theme
        )
        return {
            "id": interface.id,
            "title": interface.title,
            "description": interface.description,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/code/{interface_id}")
async def generate_gradio_code(interface_id: str) -> Dict[str, Any]:
    """Generar código de Gradio"""
    try:
        code = gradio_service.generate_gradio_code(interface_id)
        return {
            "interface_id": interface_id,
            "code": code,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




