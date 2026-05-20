"""
Projects Routes - Endpoints de gestión de proyectos
===================================================

Endpoints para gestión de proyectos siguiendo principios REST.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from ...services.project_service import ProjectService
from ...domain.models import ProjectRequest, ProjectResponse
from ...infrastructure.dependencies import get_project_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    request: ProjectRequest,
    project_service: ProjectService = Depends(get_project_service)
):
    """Crea un nuevo proyecto"""
    try:
        result = await project_service.create_project(
            description=request.description,
            project_name=request.project_name,
            author=request.author,
            version=request.version,
            priority=request.priority,
            backend_framework=request.backend_framework,
            frontend_framework=request.frontend_framework,
            generate_tests=request.generate_tests,
            include_docker=request.include_docker,
            include_docs=request.include_docs,
            tags=request.tags,
            metadata=request.metadata
        )
        return ProjectResponse(**result)
    except Exception as e:
        logger.error(f"Error creating project: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Obtiene información de un proyecto"""
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("", response_model=List[dict])
async def list_projects(
    status: Optional[str] = Query(None),
    author: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    project_service: ProjectService = Depends(get_project_service)
):
    """Lista proyectos con filtros"""
    return await project_service.list_projects(
        status=status,
        author=author,
        limit=limit,
        offset=offset
    )


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    """Elimina un proyecto"""
    success = await project_service.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")


@router.get("/queue/status")
async def get_queue_status(
    project_service: ProjectService = Depends(get_project_service)
):
    """Obtiene estado de la cola"""
    return await project_service.get_queue_status()















