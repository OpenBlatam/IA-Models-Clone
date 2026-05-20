"""
Circuit Breaker endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.circuit_breaker import CircuitBreakerService

router = APIRouter()
circuit_service = CircuitBreakerService()


@router.post("/create")
async def create_circuit(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout_seconds: int = 60
) -> Dict[str, Any]:
    """Crear circuit breaker"""
    try:
        circuit = circuit_service.create_circuit(
            name, failure_threshold, success_threshold, timeout_seconds
        )
        return {
            "name": circuit.name,
            "state": circuit.state.value,
            "failure_threshold": circuit.failure_threshold,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{circuit_name}")
async def get_circuit_status(circuit_name: str) -> Dict[str, Any]:
    """Obtener estado del circuit breaker"""
    try:
        status = circuit_service.get_circuit_status(circuit_name)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




