"""
Rutas para Automated Model Deployment
=======================================

Endpoints para despliegue automatizado.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.automated_model_deployment import (
    get_automated_model_deployment,
    AutomatedModelDeployment,
    DeploymentTarget
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/automated-model-deployment",
    tags=["Automated Model Deployment"]
)


class CreateConfigRequest(BaseModel):
    """Request para crear configuración"""
    target: str = Field("cloud", description="Target")
    scaling_config: Optional[Dict[str, Any]] = Field(None, description="Config de scaling")
    health_check_config: Optional[Dict[str, Any]] = Field(None, description="Config de health checks")


@router.post("/models/{model_id}/configs")
async def create_deployment_config(
    model_id: str,
    request: CreateConfigRequest,
    system: AutomatedModelDeployment = Depends(get_automated_model_deployment)
):
    """Crear configuración de despliegue"""
    try:
        target = DeploymentTarget(request.target)
        config = system.create_deployment_config(
            model_id,
            target,
            request.scaling_config,
            request.health_check_config
        )
        
        return {
            "config_id": config.config_id,
            "model_id": config.model_id,
            "target": config.target.value,
            "scaling_config": config.scaling_config,
            "health_check_config": config.health_check_config
        }
    except Exception as e:
        logger.error(f"Error creando configuración: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configs/{config_id}/deploy")
async def deploy_model(
    config_id: str,
    version: str = Field("1.0.0", description="Versión"),
    system: AutomatedModelDeployment = Depends(get_automated_model_deployment)
):
    """Desplegar modelo"""
    try:
        deployment = system.deploy_model(config_id, version)
        
        return {
            "deployment_id": deployment.deployment_id,
            "config_id": deployment.config.config_id,
            "status": deployment.status,
            "endpoint_url": deployment.endpoint_url,
            "version": deployment.version
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error desplegando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployments/{deployment_id}/rollback")
async def rollback_deployment(
    deployment_id: str,
    previous_version: str = Field(..., description="Versión anterior"),
    system: AutomatedModelDeployment = Depends(get_automated_model_deployment)
):
    """Hacer rollback de despliegue"""
    try:
        deployment = system.rollback_deployment(deployment_id, previous_version)
        
        return {
            "deployment_id": deployment.deployment_id,
            "status": deployment.status,
            "version": deployment.version
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error haciendo rollback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


