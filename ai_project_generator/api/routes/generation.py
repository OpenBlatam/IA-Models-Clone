"""
Generation Routes - Endpoints de generación
===========================================

Endpoints para generación de proyectos.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ...services.generation_service import GenerationService
from ...domain.models import ProjectRequest
from ...infrastructure.dependencies import get_generation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/generate", tags=["generation"])


class BatchGenerationRequest(BaseModel):
    """Request para generación en batch"""
    projects: List[ProjectRequest]
    parallel: bool = True


@router.post("")
async def generate_project(
    request: ProjectRequest,
    async_generation: bool = True,
    generation_service: GenerationService = Depends(get_generation_service)
):
    """Genera un proyecto"""
    try:
        result = await generation_service.generate_project(
            description=request.description,
            project_name=request.project_name,
            author=request.author,
            async_generation=async_generation,
            **request.dict(exclude={"description", "project_name", "author"})
        )
        return result
    except Exception as e:
        logger.error(f"Error generating project: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_generate(
    request: BatchGenerationRequest,
    generation_service: GenerationService = Depends(get_generation_service)
):
    """Genera múltiples proyectos en batch"""
    try:
        projects_data = [p.dict() for p in request.projects]
        result = await generation_service.batch_generate(
            projects=projects_data,
            parallel=request.parallel
        )
        return result
    except Exception as e:
        logger.error(f"Error in batch generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}")
async def get_generation_status(
    task_id: str,
    generation_service: GenerationService = Depends(get_generation_service)
):
    """Obtiene estado de una tarea de generación"""
    status = await generation_service.get_generation_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status















