"""
Automated Testing endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.automated_testing import AutomatedTestingService, TestType

router = APIRouter()
testing_service = AutomatedTestingService()


@router.post("/create-suite")
async def create_test_suite(name: str) -> Dict[str, Any]:
    """Crear suite de tests"""
    try:
        suite = testing_service.create_test_suite(name)
        return {
            "id": suite.id,
            "name": suite.name,
            "test_cases_count": len(suite.test_cases),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/{suite_id}")
async def run_test_suite(suite_id: str) -> Dict[str, Any]:
    """Ejecutar suite de tests"""
    try:
        results = await testing_service.run_test_suite(suite_id)
        return {
            "suite_id": suite_id,
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.status.value == "passed"),
            "failed": sum(1 for r in results if r.status.value == "failed"),
            "results": [
                {
                    "test_id": r.test_id,
                    "status": r.status.value,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                }
                for r in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{suite_id}")
async def get_test_results(suite_id: str) -> Dict[str, Any]:
    """Obtener resultados de tests"""
    try:
        results = testing_service.get_test_results(suite_id)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




