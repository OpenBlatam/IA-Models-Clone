"""
Quantum Computing API - Advanced Implementation
=============================================

Advanced quantum computing API with quantum algorithms, quantum machine learning, and quantum optimization.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import datetime

from ..services import quantum_computing_service, QuantumAlgorithm, QuantumGate, QuantumBackend

# Create router
router = APIRouter()

logger = logging.getLogger(__name__)


# Request/Response models
class QuantumBackendCreateRequest(BaseModel):
    """Quantum backend create request model"""
    name: str
    backend_type: str
    max_qubits: int
    max_operations: int = 10000
    gate_fidelity: float = 0.99
    coherence_time: float = 100.0


class QuantumCircuitCreateRequest(BaseModel):
    """Quantum circuit create request model"""
    backend_id: str
    num_qubits: int
    circuit_name: str = "Quantum Circuit"


class QuantumGateRequest(BaseModel):
    """Quantum gate request model"""
    circuit_id: str
    gate_type: str
    qubit_indices: List[int]
    parameters: Optional[Dict[str, Any]] = None


class QuantumAlgorithmRequest(BaseModel):
    """Quantum algorithm request model"""
    algorithm_type: str
    circuit_id: str
    parameters: Optional[Dict[str, Any]] = None


class QuantumBackendResponse(BaseModel):
    """Quantum backend response model"""
    backend_id: str
    name: str
    type: str
    max_qubits: int
    max_operations: int
    gate_fidelity: float
    coherence_time: float
    message: str


class QuantumCircuitResponse(BaseModel):
    """Quantum circuit response model"""
    circuit_id: str
    name: str
    backend_id: str
    num_qubits: int
    operations_count: int
    message: str


class QuantumGateResponse(BaseModel):
    """Quantum gate response model"""
    success: bool
    gate_type: str
    qubit_indices: List[int]
    message: str


class QuantumAlgorithmResponse(BaseModel):
    """Quantum algorithm response model"""
    algorithm_id: str
    algorithm_type: str
    circuit_id: str
    status: str
    message: str


class QuantumResultResponse(BaseModel):
    """Quantum result response model"""
    algorithm_id: str
    type: str
    status: str
    result: Optional[Dict[str, Any]]
    created_at: str
    completed_at: Optional[str]


class QuantumCircuitInfoResponse(BaseModel):
    """Quantum circuit info response model"""
    id: str
    name: str
    backend_id: str
    num_qubits: int
    operations_count: int
    created_at: str
    is_active: bool


class QuantumStatsResponse(BaseModel):
    """Quantum statistics response model"""
    total_circuits: int
    total_algorithms: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    algorithms_by_type: Dict[str, int]
    backends_connected: int
    total_qubits: int
    total_operations: int


# Quantum backend management endpoints
@router.post("/backends", response_model=QuantumBackendResponse)
async def create_quantum_backend(request: QuantumBackendCreateRequest):
    """Create a new quantum backend"""
    try:
        # Validate backend type
        try:
            backend_type = QuantumBackend(request.backend_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid backend type: {request.backend_type}"
            )
        
        backend_id = await quantum_computing_service.create_quantum_backend(
            name=request.name,
            backend_type=backend_type,
            max_qubits=request.max_qubits,
            max_operations=request.max_operations,
            gate_fidelity=request.gate_fidelity,
            coherence_time=request.coherence_time
        )
        
        return QuantumBackendResponse(
            backend_id=backend_id,
            name=request.name,
            type=request.backend_type,
            max_qubits=request.max_qubits,
            max_operations=request.max_operations,
            gate_fidelity=request.gate_fidelity,
            coherence_time=request.coherence_time,
            message="Quantum backend created successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create quantum backend: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create quantum backend: {str(e)}"
        )


@router.post("/circuits", response_model=QuantumCircuitResponse)
async def create_quantum_circuit(request: QuantumCircuitCreateRequest):
    """Create a quantum circuit"""
    try:
        circuit_id = await quantum_computing_service.create_quantum_circuit(
            backend_id=request.backend_id,
            num_qubits=request.num_qubits,
            circuit_name=request.circuit_name
        )
        
        return QuantumCircuitResponse(
            circuit_id=circuit_id,
            name=request.circuit_name,
            backend_id=request.backend_id,
            num_qubits=request.num_qubits,
            operations_count=0,
            message="Quantum circuit created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create quantum circuit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create quantum circuit: {str(e)}"
        )


@router.post("/gates", response_model=QuantumGateResponse)
async def add_quantum_gate(request: QuantumGateRequest):
    """Add a quantum gate to the circuit"""
    try:
        # Validate gate type
        try:
            gate_type = QuantumGate(request.gate_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid gate type: {request.gate_type}"
            )
        
        success = await quantum_computing_service.add_quantum_gate(
            circuit_id=request.circuit_id,
            gate_type=gate_type,
            qubit_indices=request.qubit_indices,
            parameters=request.parameters
        )
        
        return QuantumGateResponse(
            success=success,
            gate_type=request.gate_type,
            qubit_indices=request.qubit_indices,
            message="Quantum gate added successfully" if success else "Failed to add quantum gate"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add quantum gate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add quantum gate: {str(e)}"
        )


@router.post("/algorithms", response_model=QuantumAlgorithmResponse)
async def execute_quantum_algorithm(request: QuantumAlgorithmRequest):
    """Execute a quantum algorithm"""
    try:
        # Validate algorithm type
        try:
            algorithm_type = QuantumAlgorithm(request.algorithm_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid algorithm type: {request.algorithm_type}"
            )
        
        algorithm_id = await quantum_computing_service.execute_quantum_algorithm(
            algorithm_type=algorithm_type,
            circuit_id=request.circuit_id,
            parameters=request.parameters
        )
        
        return QuantumAlgorithmResponse(
            algorithm_id=algorithm_id,
            algorithm_type=request.algorithm_type,
            circuit_id=request.circuit_id,
            status="running",
            message="Quantum algorithm execution started successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute quantum algorithm: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute quantum algorithm: {str(e)}"
        )


# Query endpoints
@router.get("/algorithms/{algorithm_id}/result", response_model=QuantumResultResponse)
async def get_quantum_result(algorithm_id: str):
    """Get quantum algorithm result"""
    try:
        result = await quantum_computing_service.get_quantum_result(algorithm_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Algorithm not found"
            )
        
        return QuantumResultResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quantum result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quantum result: {str(e)}"
        )


@router.get("/circuits/{circuit_id}", response_model=QuantumCircuitInfoResponse)
async def get_quantum_circuit(circuit_id: str):
    """Get quantum circuit information"""
    try:
        circuit = await quantum_computing_service.get_quantum_circuit(circuit_id)
        if not circuit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Circuit not found"
            )
        
        return QuantumCircuitInfoResponse(**circuit)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quantum circuit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quantum circuit: {str(e)}"
        )


# Statistics endpoint
@router.get("/stats", response_model=QuantumStatsResponse)
async def get_quantum_stats():
    """Get quantum computing service statistics"""
    try:
        stats = await quantum_computing_service.get_quantum_stats()
        return QuantumStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get quantum stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quantum stats: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def quantum_health():
    """Quantum computing service health check"""
    try:
        stats = await quantum_computing_service.get_quantum_stats()
        
        return {
            "service": "quantum_computing_service",
            "status": "healthy",
            "backends_connected": stats["backends_connected"],
            "total_circuits": stats["total_circuits"],
            "total_algorithms": stats["total_algorithms"],
            "total_executions": stats["total_executions"],
            "successful_executions": stats["successful_executions"],
            "total_qubits": stats["total_qubits"],
            "total_operations": stats["total_operations"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Quantum computing service health check failed: {e}")
        return {
            "service": "quantum_computing_service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

