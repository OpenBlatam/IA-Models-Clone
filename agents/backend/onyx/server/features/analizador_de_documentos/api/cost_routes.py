"""
Rutas para Cost Analysis
==========================

Endpoints para análisis de costos.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.cost_analysis import (
    get_cost_analysis,
    CostAnalysis
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/cost-analysis",
    tags=["Cost Analysis"]
)


@router.get("/models/{model_id}/costs")
async def analyze_costs(
    model_id: str,
    period: str = Field("monthly", description="Período"),
    system: CostAnalysis = Depends(get_cost_analysis)
):
    """Analizar costos de modelo"""
    try:
        report = system.analyze_costs(model_id, period)
        
        return {
            "report_id": report.report_id,
            "model_id": report.model_id,
            "total_cost": report.total_cost,
            "breakdown": [
                {
                    "cost_type": cb.cost_type.value,
                    "cost_amount": cb.cost_amount,
                    "currency": cb.currency
                }
                for cb in report.breakdown
            ],
            "recommendations": report.recommendations
        }
    except Exception as e:
        logger.error(f"Error analizando costos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/estimate-training-cost")
async def estimate_training_cost(
    model_size_mb: float = Field(..., description="Tamaño del modelo en MB"),
    training_hours: float = Field(..., description="Horas de entrenamiento"),
    compute_cost_per_hour: float = Field(2.0, description="Costo por hora"),
    system: CostAnalysis = Depends(get_cost_analysis)
):
    """Estimar costo de entrenamiento"""
    try:
        cost = system.estimate_training_cost(model_size_mb, training_hours, compute_cost_per_hour)
        
        return {
            "estimated_cost": cost,
            "model_size_mb": model_size_mb,
            "training_hours": training_hours,
            "currency": "USD"
        }
    except Exception as e:
        logger.error(f"Error estimando costo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/estimate-inference-cost")
async def estimate_inference_cost(
    num_requests: int = Field(..., description="Número de requests"),
    avg_latency_ms: float = Field(..., description="Latencia promedio"),
    cost_per_1k_requests: float = Field(0.10, description="Costo por 1000 requests"),
    system: CostAnalysis = Depends(get_cost_analysis)
):
    """Estimar costo de inferencia"""
    try:
        cost = system.estimate_inference_cost(num_requests, avg_latency_ms, cost_per_1k_requests)
        
        return {
            "estimated_cost": cost,
            "num_requests": num_requests,
            "avg_latency_ms": avg_latency_ms,
            "currency": "USD"
        }
    except Exception as e:
        logger.error(f"Error estimando costo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
