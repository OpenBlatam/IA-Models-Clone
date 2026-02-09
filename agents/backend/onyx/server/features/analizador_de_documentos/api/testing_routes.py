"""
Rutas para Testing Framework
==============================

Endpoints para testing automatizado.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..core.testing_framework import get_testing_framework, TestingFramework

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/testing",
    tags=["Testing"]
)


class RegisterTestRequest(BaseModel):
    """Request para registrar test"""
    name: str = Field(..., description="Nombre del test")
    description: str = Field(..., description="Descripción")
    timeout: float = Field(30.0, description="Timeout en segundos")
    tags: Optional[List[str]] = Field(None, description="Tags")


@router.post("/register")
async def register_test(
    request: RegisterTestRequest,
    framework: TestingFramework = Depends(get_testing_framework)
):
    """Registrar nuevo test"""
    # Nota: En producción, la función de test se pasaría de otra manera
    # Por ahora, esto es solo un ejemplo
    try:
        framework.register_test(
            request.name,
            request.description,
            lambda: None,  # Placeholder
            timeout=request.timeout,
            tags=request.tags
        )
        
        return {"status": "registered", "test": request.name}
    except Exception as e:
        logger.error(f"Error registrando test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/{test_name}")
async def run_test(
    test_name: str,
    framework: TestingFramework = Depends(get_testing_framework)
):
    """Ejecutar test individual"""
    try:
        result = await framework.run_test(test_name)
        return {
            "test_name": result.test_name,
            "status": result.status.value,
            "duration": result.duration,
            "error": result.error,
            "timestamp": result.timestamp
        }
    except Exception as e:
        logger.error(f"Error ejecutando test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-all")
async def run_all_tests(
    filter_tags: Optional[List[str]] = None,
    framework: TestingFramework = Depends(get_testing_framework)
):
    """Ejecutar todos los tests"""
    try:
        results = await framework.run_all_tests(filter_tags)
        return {
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "duration": r.duration,
                    "error": r.error
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Error ejecutando tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_test_summary(
    framework: TestingFramework = Depends(get_testing_framework)
):
    """Obtener resumen de tests"""
    summary = framework.get_test_summary()
    return summary


@router.get("/report", response_class=PlainTextResponse)
async def get_test_report(
    framework: TestingFramework = Depends(get_testing_framework)
):
    """Generar reporte de tests"""
    report = framework.generate_report()
    return report
















