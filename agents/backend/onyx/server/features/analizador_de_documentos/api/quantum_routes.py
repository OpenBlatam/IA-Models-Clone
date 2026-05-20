"""
Rutas para Computación Cuántica
=================================

Endpoints para computación cuántica simulada.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.quantum_computing import get_quantum_computing, QuantumComputingSystem

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/quantum",
    tags=["Quantum Computing"]
)


class CreateCircuitRequest(BaseModel):
    """Request para crear circuito"""
    qubits: int = Field(..., ge=1, le=20, description="Número de qubits")
    circuit_id: Optional[str] = Field(None, description="ID del circuito")


class AddGateRequest(BaseModel):
    """Request para agregar puerta"""
    gate_type: str = Field(..., description="Tipo de puerta (hadamard, pauli-x, etc.)")
    qubit: int = Field(..., ge=0, description="Índice del qubit")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parámetros")


@router.post("/circuits")
async def create_circuit(
    request: CreateCircuitRequest,
    system: QuantumComputingSystem = Depends(get_quantum_computing)
):
    """Crear circuito cuántico"""
    try:
        circuit = system.create_circuit(request.qubits, request.circuit_id)
        
        return {
            "status": "created",
            "circuit_id": circuit.circuit_id,
            "qubits": circuit.qubits
        }
    except Exception as e:
        logger.error(f"Error creando circuito: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuits/{circuit_id}/gates")
async def add_gate(
    circuit_id: str,
    request: AddGateRequest,
    system: QuantumComputingSystem = Depends(get_quantum_computing)
):
    """Agregar puerta al circuito"""
    try:
        system.add_gate(
            circuit_id,
            request.gate_type,
            request.qubit,
            request.parameters
        )
        
        return {"status": "added", "circuit_id": circuit_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error agregando puerta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuits/{circuit_id}/simulate")
async def simulate_circuit(
    circuit_id: str,
    system: QuantumComputingSystem = Depends(get_quantum_computing)
):
    """Simular circuito cuántico"""
    try:
        result = system.simulate_circuit(circuit_id)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error simulando circuito: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/algorithms/grover")
async def grover_search(
    target: int = Field(..., description="Valor objetivo"),
    qubits: int = Field(4, ge=2, le=10, description="Número de qubits"),
    system: QuantumComputingSystem = Depends(get_quantum_computing)
):
    """Ejecutar algoritmo de Grover"""
    try:
        result = system.grover_search(target, qubits)
        
        return result
    except Exception as e:
        logger.error(f"Error ejecutando Grover: {e}")
        raise HTTPException(status_code=500, detail=str(e))














