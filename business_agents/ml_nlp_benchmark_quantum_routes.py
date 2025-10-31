"""
ML NLP Benchmark Quantum Computing Routes
API routes for quantum computing system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional, Union
import time
import json
import logging
from datetime import datetime

from ml_nlp_benchmark_quantum_computing import (
    get_quantum_computing,
    create_circuit,
    execute_circuit,
    create_algorithm,
    run_algorithm,
    grover_search,
    shor_factoring,
    quantum_fourier_transform,
    variational_quantum_eigensolver,
    quantum_machine_learning,
    quantum_teleportation,
    get_quantum_summary,
    clear_quantum_data
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/quantum", tags=["Quantum Computing"])

# Dependency to get quantum computing instance
def get_quantum_computing_instance():
    return get_quantum_computing()

@router.post("/circuits")
async def create_quantum_circuit(
    name: str,
    qubits: int,
    gates: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Create a quantum circuit"""
    try:
        circuit_id = create_circuit(name, qubits, gates, parameters)
        return {
            "success": True,
            "circuit_id": circuit_id,
            "message": f"Quantum circuit '{name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating quantum circuit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/circuits/{circuit_id}/execute")
async def execute_quantum_circuit(
    circuit_id: str,
    shots: int = 1000,
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Execute a quantum circuit"""
    try:
        result = execute_circuit(circuit_id, shots)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "circuit_id": result.circuit_id,
                "measurement": result.measurement,
                "probability": result.probability,
                "execution_time": result.execution_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error executing quantum circuit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/algorithms")
async def create_quantum_algorithm(
    name: str,
    algorithm_type: str,
    qubits_required: int,
    parameters: Dict[str, Any],
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Create a quantum algorithm"""
    try:
        algorithm_id = create_algorithm(name, algorithm_type, qubits_required, parameters)
        return {
            "success": True,
            "algorithm_id": algorithm_id,
            "message": f"Quantum algorithm '{name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating quantum algorithm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/algorithms/{algorithm_id}/run")
async def run_quantum_algorithm(
    algorithm_id: str,
    input_data: Any,
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Run a quantum algorithm"""
    try:
        result = run_algorithm(algorithm_id, input_data)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "circuit_id": result.circuit_id,
                "measurement": result.measurement,
                "probability": result.probability,
                "execution_time": result.execution_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running quantum algorithm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/grover-search")
async def run_grover_search(
    search_space: List[Any],
    target: Any,
    iterations: Optional[int] = None,
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Run Grover's search algorithm"""
    try:
        result = grover_search(search_space, target, iterations)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "circuit_id": result.circuit_id,
                "measurement": result.measurement,
                "probability": result.probability,
                "execution_time": result.execution_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running Grover search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shor-factoring")
async def run_shor_factoring(
    number: int,
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Run Shor's factoring algorithm"""
    try:
        result = shor_factoring(number)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "circuit_id": result.circuit_id,
                "measurement": result.measurement,
                "probability": result.probability,
                "execution_time": result.execution_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running Shor factoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quantum-fourier-transform")
async def run_quantum_fourier_transform(
    input_state: List[complex],
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Run Quantum Fourier Transform"""
    try:
        result = quantum_fourier_transform(input_state)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "circuit_id": result.circuit_id,
                "measurement": result.measurement,
                "probability": result.probability,
                "execution_time": result.execution_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running Quantum Fourier Transform: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/variational-quantum-eigensolver")
async def run_variational_quantum_eigensolver(
    hamiltonian: List[List[float]],
    ansatz: List[Dict[str, Any]],
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Run Variational Quantum Eigensolver"""
    try:
        import numpy as np
        hamiltonian_array = np.array(hamiltonian)
        result = variational_quantum_eigensolver(hamiltonian_array, ansatz)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "circuit_id": result.circuit_id,
                "measurement": result.measurement,
                "probability": result.probability,
                "execution_time": result.execution_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running Variational Quantum Eigensolver: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quantum-machine-learning")
async def run_quantum_machine_learning(
    training_data: List[Dict[str, Any]],
    model_type: str = "classification",
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Run Quantum Machine Learning"""
    try:
        result = quantum_machine_learning(training_data, model_type)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "circuit_id": result.circuit_id,
                "measurement": result.measurement,
                "probability": result.probability,
                "execution_time": result.execution_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running Quantum Machine Learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quantum-teleportation")
async def run_quantum_teleportation(
    qubit_state: List[complex],
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Run Quantum Teleportation"""
    try:
        result = quantum_teleportation(qubit_state)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "circuit_id": result.circuit_id,
                "measurement": result.measurement,
                "probability": result.probability,
                "execution_time": result.execution_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running Quantum Teleportation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/circuits")
async def list_quantum_circuits(
    active_only: bool = False,
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """List quantum circuits"""
    try:
        circuits = quantum_computing.list_circuits(active_only=active_only)
        return {
            "success": True,
            "circuits": [
                {
                    "circuit_id": circuit.circuit_id,
                    "name": circuit.name,
                    "qubits": circuit.qubits,
                    "gates": circuit.gates,
                    "parameters": circuit.parameters,
                    "created_at": circuit.created_at.isoformat(),
                    "is_active": circuit.is_active
                }
                for circuit in circuits
            ]
        }
    except Exception as e:
        logger.error(f"Error listing quantum circuits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/circuits/{circuit_id}")
async def get_quantum_circuit(
    circuit_id: str,
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Get quantum circuit information"""
    try:
        circuit = quantum_computing.get_circuit(circuit_id)
        if not circuit:
            raise HTTPException(status_code=404, detail="Circuit not found")
        
        return {
            "success": True,
            "circuit": {
                "circuit_id": circuit.circuit_id,
                "name": circuit.name,
                "qubits": circuit.qubits,
                "gates": circuit.gates,
                "parameters": circuit.parameters,
                "created_at": circuit.created_at.isoformat(),
                "last_updated": circuit.last_updated.isoformat(),
                "is_active": circuit.is_active,
                "metadata": circuit.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quantum circuit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/algorithms")
async def list_quantum_algorithms(
    algorithm_type: Optional[str] = None,
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """List quantum algorithms"""
    try:
        algorithms = quantum_computing.list_algorithms(algorithm_type=algorithm_type)
        return {
            "success": True,
            "algorithms": [
                {
                    "algorithm_id": algorithm.algorithm_id,
                    "name": algorithm.name,
                    "algorithm_type": algorithm.algorithm_type,
                    "qubits_required": algorithm.qubits_required,
                    "parameters": algorithm.parameters,
                    "complexity": algorithm.complexity,
                    "created_at": algorithm.created_at.isoformat(),
                    "is_implemented": algorithm.is_implemented
                }
                for algorithm in algorithms
            ]
        }
    except Exception as e:
        logger.error(f"Error listing quantum algorithms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/algorithms/{algorithm_id}")
async def get_quantum_algorithm(
    algorithm_id: str,
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Get quantum algorithm information"""
    try:
        algorithm = quantum_computing.get_algorithm(algorithm_id)
        if not algorithm:
            raise HTTPException(status_code=404, detail="Algorithm not found")
        
        return {
            "success": True,
            "algorithm": {
                "algorithm_id": algorithm.algorithm_id,
                "name": algorithm.name,
                "algorithm_type": algorithm.algorithm_type,
                "qubits_required": algorithm.qubits_required,
                "parameters": algorithm.parameters,
                "complexity": algorithm.complexity,
                "created_at": algorithm.created_at.isoformat(),
                "last_updated": algorithm.last_updated.isoformat(),
                "is_implemented": algorithm.is_implemented,
                "metadata": algorithm.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quantum algorithm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results")
async def get_quantum_results(
    circuit_id: Optional[str] = None,
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Get quantum results"""
    try:
        results = quantum_computing.get_quantum_results(circuit_id=circuit_id)
        return {
            "success": True,
            "results": [
                {
                    "result_id": result.result_id,
                    "circuit_id": result.circuit_id,
                    "measurement": result.measurement,
                    "probability": result.probability,
                    "execution_time": result.execution_time,
                    "success": result.success,
                    "error_message": result.error_message,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error getting quantum results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_quantum_summary_endpoint(
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Get quantum computing system summary"""
    try:
        summary = get_quantum_summary()
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error getting quantum summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear")
async def clear_quantum_data_endpoint(
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Clear all quantum computing data"""
    try:
        clear_quantum_data()
        return {
            "success": True,
            "message": "All quantum computing data cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing quantum data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_quantum_capabilities(
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Get quantum computing capabilities"""
    try:
        capabilities = quantum_computing.quantum_capabilities
        gates = list(quantum_computing.quantum_gates.keys())
        algorithms = list(quantum_computing.quantum_algorithms.keys())
        states = list(quantum_computing.quantum_states.keys())
        measurement_bases = list(quantum_computing.measurement_bases.keys())
        
        return {
            "success": True,
            "capabilities": {
                "quantum_capabilities": capabilities,
                "quantum_gates": gates,
                "quantum_algorithms": algorithms,
                "quantum_states": states,
                "measurement_bases": measurement_bases
            }
        }
    except Exception as e:
        logger.error(f"Error getting quantum capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def quantum_health_check(
    quantum_computing = Depends(get_quantum_computing_instance)
):
    """Quantum computing system health check"""
    try:
        summary = get_quantum_summary()
        health_status = "healthy" if summary["total_circuits"] >= 0 else "unhealthy"
        
        return {
            "success": True,
            "health": {
                "status": health_status,
                "total_circuits": summary["total_circuits"],
                "total_algorithms": summary["total_algorithms"],
                "total_results": summary["total_results"],
                "active_circuits": summary["active_circuits"],
                "implemented_algorithms": summary["implemented_algorithms"]
            }
        }
    except Exception as e:
        logger.error(f"Error in quantum health check: {e}")
        return {
            "success": False,
            "health": {
                "status": "unhealthy",
                "error": str(e)
            }
        }











