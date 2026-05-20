"""
Rutas para Model Governance
=============================

Endpoints para gobernanza de modelos.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_governance import (
    get_model_governance,
    ModelGovernance
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/model-governance",
    tags=["Model Governance"]
)


class CreatePolicyRequest(BaseModel):
    """Request para crear política"""
    name: str = Field(..., description="Nombre")
    description: str = Field(..., description="Descripción")
    rules: List[Dict[str, Any]] = Field(..., description="Reglas")


class RequestApprovalRequest(BaseModel):
    """Request para solicitar aprobación"""
    requirements: List[str] = Field(..., description="Requisitos")


@router.post("/policies")
async def create_policy(
    request: CreatePolicyRequest,
    system: ModelGovernance = Depends(get_model_governance)
):
    """Crear política de gobernanza"""
    try:
        policy = system.create_policy(
            request.name,
            request.description,
            request.rules
        )
        
        return {
            "policy_id": policy.policy_id,
            "name": policy.name,
            "description": policy.description,
            "rules": policy.rules
        }
    except Exception as e:
        logger.error(f"Error creando política: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/approvals")
async def request_approval(
    model_id: str,
    request: RequestApprovalRequest,
    system: ModelGovernance = Depends(get_model_governance)
):
    """Solicitar aprobación de modelo"""
    try:
        approval = system.request_approval(model_id, request.requirements)
        
        return {
            "approval_id": approval.approval_id,
            "model_id": approval.model_id,
            "status": approval.status.value,
            "requirements": approval.requirements
        }
    except Exception as e:
        logger.error(f"Error solicitando aprobación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approvals/{approval_id}/approve")
async def approve_model(
    approval_id: str,
    approver: str = Field(..., description="Aprobador"),
    comments: str = Field("", description="Comentarios"),
    system: ModelGovernance = Depends(get_model_governance)
):
    """Aprobar modelo"""
    try:
        approval = system.approve_model(approval_id, approver, comments)
        
        return {
            "approval_id": approval.approval_id,
            "model_id": approval.model_id,
            "status": approval.status.value,
            "approver": approval.approver,
            "comments": approval.comments
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error aprobando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/check-compliance")
async def check_compliance(
    model_id: str,
    policy_id: str = Field(..., description="ID de política"),
    system: ModelGovernance = Depends(get_model_governance)
):
    """Verificar compliance"""
    try:
        compliance = system.check_compliance(model_id, policy_id)
        
        return compliance
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error verificando compliance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


