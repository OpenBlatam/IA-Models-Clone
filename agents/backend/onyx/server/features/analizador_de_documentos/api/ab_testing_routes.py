"""
Rutas para A/B Testing
=======================

Endpoints para tests A/B de modelos.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.ab_testing import get_ab_testing_manager, ABTestingManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/ab-testing",
    tags=["A/B Testing"]
)


class VariantConfig(BaseModel):
    """Configuración de variante"""
    variant_id: str = Field(..., description="ID de variante")
    model_config: Dict[str, Any] = Field(..., description="Configuración del modelo")
    traffic_percentage: float = Field(50.0, description="Porcentaje de tráfico")


class CreateTestRequest(BaseModel):
    """Request para crear test"""
    test_id: str = Field(..., description="ID del test")
    name: str = Field(..., description="Nombre del test")
    variants: List[VariantConfig] = Field(..., description="Variantes")


@router.post("/tests")
async def create_test(
    request: CreateTestRequest,
    manager: ABTestingManager = Depends(get_ab_testing_manager)
):
    """Crear nuevo test A/B"""
    try:
        test = manager.create_test(
            request.test_id,
            request.name,
            [v.dict() for v in request.variants]
        )
        
        return {
            "status": "created",
            "test_id": test.test_id,
            "name": test.name,
            "status": test.status.value
        }
    except Exception as e:
        logger.error(f"Error creando test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tests/{test_id}/variant")
async def get_variant(
    test_id: str,
    manager: ABTestingManager = Depends(get_ab_testing_manager)
):
    """Obtener variante para request"""
    variant_id = manager.select_variant(test_id)
    
    if variant_id is None:
        raise HTTPException(status_code=404, detail="Test no encontrado o inactivo")
    
    return {"test_id": test_id, "variant_id": variant_id}


@router.post("/tests/{test_id}/result")
async def record_result(
    test_id: str,
    variant_id: str,
    success: bool,
    latency: float,
    accuracy: float = None,
    manager: ABTestingManager = Depends(get_ab_testing_manager)
):
    """Registrar resultado de request"""
    try:
        manager.record_result(test_id, variant_id, success, latency, accuracy)
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Error registrando resultado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tests/{test_id}/results")
async def get_test_results(
    test_id: str,
    manager: ABTestingManager = Depends(get_ab_testing_manager)
):
    """Obtener resultados del test"""
    results = manager.get_test_results(test_id)
    
    if not results:
        raise HTTPException(status_code=404, detail="Test no encontrado")
    
    return results


@router.post("/tests/{test_id}/stop")
async def stop_test(
    test_id: str,
    manager: ABTestingManager = Depends(get_ab_testing_manager)
):
    """Detener test"""
    try:
        manager.stop_test(test_id)
        return {"status": "stopped", "test_id": test_id}
    except Exception as e:
        logger.error(f"Error deteniendo test: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















