"""
Export Routes - Endpoints de exportación
========================================

Endpoints para exportación de proyectos.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ...services.export_service import ExportService
from ...infrastructure.dependencies import get_export_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/export", tags=["export"])


class ExportRequest(BaseModel):
    """Request para exportación"""
    project_path: str
    format: str = "zip"  # zip, tar, tar.gz


@router.post("/zip")
async def export_zip(
    request: ExportRequest,
    export_service: ExportService = Depends(get_export_service)
):
    """Exporta proyecto a ZIP"""
    try:
        result = await export_service.export_project(
            request.project_path,
            format="zip"
        )
        return result
    except Exception as e:
        logger.error(f"Error exporting project: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))















