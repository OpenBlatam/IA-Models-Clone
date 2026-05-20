"""
API Documentation endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.api_documentation import APIDocumentationService

router = APIRouter()
docs_service = APIDocumentationService()


@router.post("/generate")
async def generate_documentation(
    version: str,
    title: str,
    description: str,
    endpoints: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generar documentación de API"""
    try:
        doc = docs_service.generate_documentation(version, title, description, endpoints)
        return {
            "version": doc.version,
            "title": doc.title,
            "endpoints_count": len(doc.endpoints),
            "generated_at": doc.generated_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/openapi/{version}")
async def get_openapi_spec(version: str) -> Dict[str, Any]:
    """Obtener especificación OpenAPI"""
    try:
        spec = docs_service.generate_openapi_spec(version)
        return spec
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{version}")
async def export_documentation(
    version: str,
    format: str = "json"
) -> Dict[str, Any]:
    """Exportar documentación"""
    try:
        exported = docs_service.export_documentation(version, format)
        return {
            "version": version,
            "format": format,
            "content": exported,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




