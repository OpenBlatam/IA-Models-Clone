"""
Validation Routes - Endpoints de validación
===========================================

Endpoints para validación de proyectos.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ...services.validation_service import ValidationService
from ...infrastructure.dependencies import get_validation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/validate", tags=["validation"])


class ValidationRequest(BaseModel):
    """Request para validación"""
    project_path: str


@router.post("")
async def validate_project(
    request: ValidationRequest,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """Valida un proyecto"""
    try:
        result = await validation_service.validate_project(request.project_path)
        return result
    except Exception as e:
        logger.error(f"Error validating project: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))















