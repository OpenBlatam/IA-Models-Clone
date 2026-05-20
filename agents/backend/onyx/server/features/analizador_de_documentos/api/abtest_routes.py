"""
Rutas para Advanced A/B Testing
==================================

Endpoints para A/B testing avanzado.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_ab_testing import (
    get_ab_testing,
    AdvancedABTesting
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/ab-testing",
    tags=["Advanced A/B Testing"]
)


class CreateTestRequest(BaseModel):
    """Request para crear test"""
    variant_a: str = Field(..., description="Modelo A")
    variant_b: str = Field(..., description="Modelo B")
    traffic_split: float = Field(0.5, description="Split de tráfico")


@router.post("/tests")
async def create_test(
    request: CreateTestRequest,
    system: AdvancedABTesting = Depends(get_ab_testing)
):
    """Crear A/B test"""
    try:
        test = system.create_test(
            request.variant_a,
            request.variant_b,
            request.traffic_split
        )
        
        return {
            "test_id": test.test_id,
            "variant_a": test.variant_a,
            "variant_b": test.variant_b,
            "traffic_split": test.traffic_split,
            "status": test.status.value
        }
    except Exception as e:
        logger.error(f"Error creando test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tests/{test_id}/start")
async def start_test(
    test_id: str,
    system: AdvancedABTesting = Depends(get_ab_testing)
):
    """Iniciar test"""
    try:
        system.start_test(test_id)
        
        return {"status": "started", "test_id": test_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error iniciando test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tests/{test_id}/analyze")
async def analyze_results(
    test_id: str,
    min_samples: int = Field(100, description="Mínimo de muestras"),
    system: AdvancedABTesting = Depends(get_ab_testing)
):
    """Analizar resultados"""
    try:
        result = system.analyze_results(test_id, min_samples)
        
        return {
            "test_id": result.test_id,
            "variant_a_metrics": result.variant_a_metrics,
            "variant_b_metrics": result.variant_b_metrics,
            "winner": result.winner,
            "confidence": result.confidence
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analizando resultados: {e}")
        raise HTTPException(status_code=500, detail=str(e))


