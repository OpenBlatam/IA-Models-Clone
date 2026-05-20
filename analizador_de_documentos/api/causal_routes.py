"""
Rutas para Causal Inference
=============================

Endpoints para inferencia causal.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.causal_inference import (
    get_causal_inference,
    CausalInference,
    CausalMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/causal-inference",
    tags=["Causal Inference"]
)


class EstimateEffectRequest(BaseModel):
    """Request para estimar efecto"""
    treatment: str = Field(..., description="Variable de tratamiento")
    outcome: str = Field(..., description="Variable de resultado")
    data: List[Dict[str, Any]] = Field(..., description="Datos")
    method: str = Field("propensity_score", description="Método")


@router.post("/estimate-effect")
async def estimate_effect(
    request: EstimateEffectRequest,
    system: CausalInference = Depends(get_causal_inference)
):
    """Estimar efecto causal"""
    try:
        method = CausalMethod(request.method)
        effect = system.estimate_effect(
            request.treatment,
            request.outcome,
            request.data,
            method
        )
        
        return {
            "treatment": effect.treatment,
            "outcome": effect.outcome,
            "effect_size": effect.effect_size,
            "confidence_interval": effect.confidence_interval,
            "p_value": effect.p_value,
            "method": effect.method.value
        }
    except Exception as e:
        logger.error(f"Error estimando efecto: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/identify-confounders")
async def identify_confounders(
    treatment: str = Field(..., description="Tratamiento"),
    outcome: str = Field(..., description="Resultado"),
    variables: List[str] = Field(..., description="Variables"),
    system: CausalInference = Depends(get_causal_inference)
):
    """Identificar confounders"""
    try:
        confounders = system.identify_confounders(treatment, outcome, variables)
        
        return {"confounders": confounders, "count": len(confounders)}
    except Exception as e:
        logger.error(f"Error identificando confounders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-assumptions")
async def validate_assumptions(
    treatment: str = Field(..., description="Tratamiento"),
    outcome: str = Field(..., description="Resultado"),
    effect_size: float = Field(..., description="Tamaño del efecto"),
    system: CausalInference = Depends(get_causal_inference)
):
    """Validar supuestos causales"""
    try:
        # Crear efecto temporal para validación
        from ..core.causal_inference import CausalEffect, CausalMethod
        effect = CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=effect_size,
            confidence_interval={"lower": effect_size - 0.05, "upper": effect_size + 0.05},
            p_value=0.001,
            method=CausalMethod.PROPENSITY_SCORE
        )
        
        validation = system.validate_causal_assumptions(effect)
        
        return validation
    except Exception as e:
        logger.error(f"Error validando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


