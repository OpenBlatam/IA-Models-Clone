"""
Quantum Computing Routes for Email Sequence System

This module provides API endpoints for quantum computing capabilities
including quantum algorithms and quantum machine learning.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from .schemas import ErrorResponse
from ..core.quantum_computing_engine import (
    quantum_computing_engine,
    QuantumAlgorithm,
    QuantumBackend
)
from ..core.dependencies import get_current_user
from ..core.exceptions import QuantumComputingError

logger = logging.getLogger(__name__)

# Quantum Computing router
quantum_computing_router = APIRouter(
    prefix="/api/v1/quantum-computing",
    tags=["Quantum Computing"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


@quantum_computing_router.post("/circuits")
async def create_quantum_circuit(
    circuit_id: str,
    qubits: int,
    gates: List[Dict[str, Any]],
    measurements: Optional[List[int]] = None
):
    """
    Create a quantum circuit.
    
    Args:
        circuit_id: Unique circuit identifier
        qubits: Number of qubits
        gates: List of quantum gates
        measurements: Qubits to measure
        
    Returns:
        Circuit creation result
    """
    try:
        circuit = await quantum_computing_engine.create_quantum_circuit(
            circuit_id=circuit_id,
            qubits=qubits,
            gates=gates,
            measurements=measurements
        )
        
        return {
            "status": "success",
            "circuit_id": circuit.circuit_id,
            "qubits": circuit.qubits,
            "depth": circuit.depth,
            "gates": len(circuit.gates),
            "measurements": circuit.measurements,
            "created_at": circuit.created_at.isoformat(),
            "message": "Quantum circuit created successfully"
        }
        
    except QuantumComputingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating quantum circuit: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.post("/tasks/grover-search")
async def execute_grover_search(
    search_space: List[Any],
    target_element: Any,
    backend: QuantumBackend = QuantumBackend.SIMULATOR
):
    """
    Execute Grover's search algorithm.
    
    Args:
        search_space: List of elements to search
        target_element: Element to find
        backend: Quantum backend to use
        
    Returns:
        Task submission result
    """
    try:
        task_id = await quantum_computing_engine.execute_grover_search(
            search_space=search_space,
            target_element=target_element,
            backend=backend
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "algorithm": "grover_search",
            "search_space_size": len(search_space),
            "target_element": target_element,
            "backend": backend.value,
            "message": "Grover search task submitted successfully"
        }
        
    except QuantumComputingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing Grover search: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.post("/tasks/quantum-optimization")
async def execute_quantum_optimization(
    optimization_problem: Dict[str, Any],
    backend: QuantumBackend = QuantumBackend.SIMULATOR
):
    """
    Execute quantum optimization using QAOA.
    
    Args:
        optimization_problem: Optimization problem definition
        backend: Quantum backend to use
        
    Returns:
        Task submission result
    """
    try:
        task_id = await quantum_computing_engine.execute_quantum_optimization(
            optimization_problem=optimization_problem,
            backend=backend
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "algorithm": "qaoa_optimization",
            "problem_type": optimization_problem.get("type", "max_cut"),
            "variables": optimization_problem.get("variables", 4),
            "backend": backend.value,
            "message": "Quantum optimization task submitted successfully"
        }
        
    except QuantumComputingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing quantum optimization: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.post("/tasks/quantum-ml")
async def execute_quantum_ml(
    ml_problem: Dict[str, Any],
    backend: QuantumBackend = QuantumBackend.SIMULATOR
):
    """
    Execute quantum machine learning algorithm.
    
    Args:
        ml_problem: Machine learning problem definition
        backend: Quantum backend to use
        
    Returns:
        Task submission result
    """
    try:
        task_id = await quantum_computing_engine.execute_quantum_ml(
            ml_problem=ml_problem,
            backend=backend
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "algorithm": "quantum_ml",
            "problem_type": ml_problem.get("type", "classification"),
            "feature_qubits": ml_problem.get("feature_qubits", 4),
            "backend": backend.value,
            "message": "Quantum ML task submitted successfully"
        }
        
    except QuantumComputingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing quantum ML: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.post("/tasks/generic")
async def submit_generic_quantum_task(
    algorithm: QuantumAlgorithm,
    backend: QuantumBackend,
    circuit_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    priority: int = 1
):
    """
    Submit a generic quantum computing task.
    
    Args:
        algorithm: Quantum algorithm to execute
        backend: Quantum backend to use
        circuit_id: Quantum circuit ID
        parameters: Algorithm parameters
        priority: Task priority
        
    Returns:
        Task submission result
    """
    try:
        if circuit_id not in quantum_computing_engine.circuits:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quantum circuit not found")
        
        circuit = quantum_computing_engine.circuits[circuit_id]
        
        task_id = await quantum_computing_engine.submit_quantum_task(
            algorithm=algorithm,
            backend=backend,
            circuit=circuit,
            parameters=parameters,
            priority=priority
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "algorithm": algorithm.value,
            "backend": backend.value,
            "circuit_id": circuit_id,
            "priority": priority,
            "message": "Generic quantum task submitted successfully"
        }
        
    except QuantumComputingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting generic quantum task: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.get("/tasks/{task_id}")
async def get_quantum_task_result(task_id: str):
    """
    Get quantum task result.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task result
    """
    try:
        result = await quantum_computing_engine.get_quantum_task_result(task_id)
        return result
        
    except QuantumComputingError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting quantum task result: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.get("/tasks")
async def list_quantum_tasks():
    """
    List all quantum computing tasks.
    
    Returns:
        List of tasks
    """
    try:
        tasks = []
        for task_id, task in quantum_computing_engine.tasks.items():
            tasks.append({
                "task_id": task_id,
                "algorithm": task.algorithm.value,
                "backend": task.backend.value,
                "status": task.status.value,
                "priority": task.priority,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "circuit_id": task.circuit.circuit_id,
                "qubits": task.circuit.qubits,
                "depth": task.circuit.depth
            })
        
        return {
            "status": "success",
            "tasks": tasks,
            "total_tasks": len(tasks)
        }
        
    except Exception as e:
        logger.error(f"Error listing quantum tasks: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.get("/circuits")
async def list_quantum_circuits():
    """
    List all quantum circuits.
    
    Returns:
        List of circuits
    """
    try:
        circuits = []
        for circuit_id, circuit in quantum_computing_engine.circuits.items():
            circuits.append({
                "circuit_id": circuit_id,
                "qubits": circuit.qubits,
                "depth": circuit.depth,
                "gates": len(circuit.gates),
                "measurements": circuit.measurements,
                "created_at": circuit.created_at.isoformat(),
                "metadata": circuit.metadata
            })
        
        return {
            "status": "success",
            "circuits": circuits,
            "total_circuits": len(circuits)
        }
        
    except Exception as e:
        logger.error(f"Error listing quantum circuits: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.get("/backends")
async def list_quantum_backends():
    """
    List available quantum computing backends.
    
    Returns:
        List of backends
    """
    try:
        backends = []
        for backend_type, backend_info in quantum_computing_engine.backends.items():
            backends.append({
                "backend": backend_type.value,
                "name": backend_info["name"],
                "qubits": backend_info["qubits"],
                "available": backend_info["available"],
                "queue_time": backend_info["queue_time"],
                "cost_per_shot": backend_info["cost_per_shot"]
            })
        
        return {
            "status": "success",
            "backends": backends,
            "total_backends": len(backends)
        }
        
    except Exception as e:
        logger.error(f"Error listing quantum backends: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.get("/stats")
async def get_quantum_computing_stats():
    """
    Get quantum computing engine statistics.
    
    Returns:
        Engine statistics
    """
    try:
        stats = await quantum_computing_engine.get_quantum_engine_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting quantum computing stats: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.delete("/circuits/{circuit_id}")
async def delete_quantum_circuit(circuit_id: str):
    """
    Delete a quantum circuit.
    
    Args:
        circuit_id: Circuit ID to delete
        
    Returns:
        Deletion result
    """
    try:
        if circuit_id not in quantum_computing_engine.circuits:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quantum circuit not found")
        
        # Remove circuit
        del quantum_computing_engine.circuits[circuit_id]
        
        # Remove from cache
        await quantum_computing_engine.cache_manager.delete(f"quantum_circuit:{circuit_id}")
        
        return {
            "status": "success",
            "circuit_id": circuit_id,
            "message": "Quantum circuit deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting quantum circuit: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@quantum_computing_router.post("/tasks/{task_id}/cancel")
async def cancel_quantum_task(task_id: str):
    """
    Cancel a quantum computing task.
    
    Args:
        task_id: Task ID to cancel
        
    Returns:
        Cancellation result
    """
    try:
        if task_id not in quantum_computing_engine.tasks:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quantum task not found")
        
        task = quantum_computing_engine.tasks[task_id]
        
        if task.status.value in ["completed", "failed", "cancelled"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Task cannot be cancelled")
        
        task.status = "cancelled"
        task.completed_at = datetime.utcnow()
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Quantum task cancelled successfully"
        }
        
    except Exception as e:
        logger.error(f"Error cancelling quantum task: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


# Error handlers for quantum computing routes
@quantum_computing_router.exception_handler(QuantumComputingError)
async def quantum_computing_error_handler(request, exc):
    """Handle quantum computing errors"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=f"Quantum computing error: {exc.message}",
            error_code="QUANTUM_COMPUTING_ERROR"
        ).dict()
    )






























