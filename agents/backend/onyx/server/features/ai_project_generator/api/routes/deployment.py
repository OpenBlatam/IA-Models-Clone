"""
Deployment Routes - Endpoints de despliegue
===========================================

Endpoints para despliegue de proyectos.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ...services.deployment_service import DeploymentService
from ...infrastructure.dependencies import get_deployment_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/deploy", tags=["deployment"])


class DeploymentRequest(BaseModel):
    """Request para despliegue"""
    project_path: str
    platform: str  # vercel, netlify, railway, heroku


@router.post("")
async def deploy_project(
    request: DeploymentRequest,
    deployment_service: DeploymentService = Depends(get_deployment_service)
):
    """Despliega un proyecto"""
    try:
        result = await deployment_service.deploy(
            request.project_path,
            request.platform
        )
        return result
    except Exception as e:
        logger.error(f"Error deploying project: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))















