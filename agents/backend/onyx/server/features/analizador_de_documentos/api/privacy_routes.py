"""
Rutas para Differential Privacy
=================================

Endpoints para privacidad diferencial.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.differential_privacy import (
    get_differential_privacy,
    DifferentialPrivacy,
    PrivacyMechanism
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/differential-privacy",
    tags=["Differential Privacy"]
)


class ApplyPrivacyRequest(BaseModel):
    """Request para aplicar privacidad"""
    data: List[Dict[str, Any]] = Field(..., description="Datos")
    sensitive_fields: List[str] = Field(..., description="Campos sensibles")


@router.post("/create-config")
async def create_privacy_config(
    epsilon: float = Field(..., description="Privacy budget"),
    delta: float = Field(0.0, description="Delta"),
    mechanism: str = Field("laplace", description="Mecanismo"),
    system: DifferentialPrivacy = Depends(get_differential_privacy)
):
    """Crear configuración de privacidad"""
    try:
        mechanism_enum = PrivacyMechanism(mechanism)
        config = system.create_privacy_config(epsilon, delta, mechanism_enum)
        
        return {
            "epsilon": config.epsilon,
            "delta": config.delta,
            "mechanism": config.mechanism.value
        }
    except Exception as e:
        logger.error(f"Error creando configuración: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply")
async def apply_differential_privacy(
    epsilon: float = Field(..., description="Epsilon"),
    delta: float = Field(0.0, description="Delta"),
    mechanism: str = Field("laplace", description="Mecanismo"),
    request: ApplyPrivacyRequest = ...,
    system: DifferentialPrivacy = Depends(get_differential_privacy)
):
    """Aplicar privacidad diferencial"""
    try:
        from ..core.differential_privacy import PrivacyConfig, PrivacyMechanism
        
        mechanism_enum = PrivacyMechanism(mechanism)
        config = PrivacyConfig(
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism_enum
        )
        
        result = system.apply_differential_privacy(
            request.data,
            config,
            request.sensitive_fields
        )
        
        return result
    except Exception as e:
        logger.error(f"Error aplicando privacidad: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-budget")
async def calculate_budget(
    operations: List[Dict[str, Any]] = Field(..., description="Operaciones"),
    system: DifferentialPrivacy = Depends(get_differential_privacy)
):
    """Calcular budget de privacidad"""
    try:
        budget = system.calculate_privacy_budget(operations)
        
        return {"total_budget": budget, "operations": len(operations)}
    except Exception as e:
        logger.error(f"Error calculando budget: {e}")
        raise HTTPException(status_code=500, detail=str(e))


