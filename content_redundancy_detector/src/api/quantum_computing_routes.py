"""
Quantum Computing API Routes - Advanced quantum computing endpoints
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np

from ..core.quantum_computing_engine import (
    get_quantum_computing_engine, QuantumConfig, 
    QuantumCircuit, QuantumAlgorithm, QuantumState
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quantum-computing", tags=["Quantum Computing"])


# Request/Response Models
class QuantumCircuitRequest(BaseModel):
    """Quantum circuit creation request model"""
    qubits: int = Field(..., description="Number of qubits", ge=1, le=32)
    backend: str = Field(default="qiskit", description="Quantum backend (qiskit, cirq, pennylane)")
    name: str = Field(default="quantum_circuit", description="Circuit name")
    gates: Optional[List[Dict[str, Any]]] = Field(default=None, description="Custom gates to add")
    measurements: Optional[List[Dict[str, Any]]] = Field(default=None, description="Custom measurements")


class QuantumExecutionRequest(BaseModel):
    """Quantum circuit execution request model"""
    circuit_id: str = Field(..., description="Circuit ID to execute", min_length=1)
    backend: str = Field(default="qiskit", description="Quantum backend (qiskit, cirq, pennylane)")
    shots: int = Field(default=1024, description="Number of shots", ge=1, le=10000)
    optimization_level: int = Field(default=3, description="Optimization level", ge=0, le=3)


class QuantumAlgorithmRequest(BaseModel):
    """Quantum algorithm execution request model"""
    algorithm_type: str = Field(..., description="Algorithm type (grover_search, quantum_fourier_transform, etc.)")
    parameters: Dict[str, Any] = Field(..., description="Algorithm parameters")
    backend: str = Field(default="qiskit", description="Quantum backend")


class QuantumConfigRequest(BaseModel):
    """Quantum computing configuration request model"""
    enable_qiskit: bool = Field(default=True, description="Enable Qiskit backend")
    enable_cirq: bool = Field(default=True, description="Enable Cirq backend")
    enable_pennylane: bool = Field(default=True, description="Enable PennyLane backend")
    enable_ibm_quantum: bool = Field(default=False, description="Enable IBM Quantum")
    enable_google_quantum: bool = Field(default=False, description="Enable Google Quantum")
    enable_ionq: bool = Field(default=False, description="Enable IonQ")
    enable_rigetti: bool = Field(default=False, description="Enable Rigetti")
    max_qubits: int = Field(default=32, description="Maximum number of qubits", ge=1, le=100)
    max_depth: int = Field(default=100, description="Maximum circuit depth", ge=1, le=1000)
    optimization_level: int = Field(default=3, description="Optimization level", ge=0, le=3)
    enable_error_mitigation: bool = Field(default=True, description="Enable error mitigation")
    enable_quantum_ml: bool = Field(default=True, description="Enable quantum machine learning")
    enable_quantum_optimization: bool = Field(default=True, description="Enable quantum optimization")
    enable_quantum_cryptography: bool = Field(default=True, description="Enable quantum cryptography")
    enable_quantum_simulation: bool = Field(default=True, description="Enable quantum simulation")
    backend_type: str = Field(default="simulator", description="Backend type")
    shots: int = Field(default=1024, description="Default number of shots", ge=1, le=10000)


# Dependency to get quantum computing engine
async def get_quantum_engine():
    """Get quantum computing engine dependency"""
    engine = await get_quantum_computing_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Quantum Computing Engine not available")
    return engine


# Quantum Computing Routes
@router.post("/create-circuit", response_model=Dict[str, Any])
async def create_quantum_circuit(
    request: QuantumCircuitRequest,
    engine: QuantumComputingEngine = Depends(get_quantum_engine)
):
    """Create a quantum circuit"""
    try:
        start_time = time.time()
        
        # Create quantum circuit
        circuit = await engine.create_quantum_circuit(
            qubits=request.qubits,
            backend=request.backend,
            name=request.name
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "quantum_circuit": {
                "circuit_id": circuit.circuit_id,
                "timestamp": circuit.timestamp.isoformat(),
                "qubits": circuit.qubits,
                "depth": circuit.depth,
                "gates": circuit.gates,
                "measurements": circuit.measurements,
                "backend": circuit.backend,
                "status": circuit.status
            },
            "processing_time_ms": processing_time,
            "message": f"Quantum circuit created successfully with {request.qubits} qubits"
        }
        
    except Exception as e:
        logger.error(f"Error creating quantum circuit: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit creation failed: {str(e)}")


@router.post("/execute-circuit", response_model=Dict[str, Any])
async def execute_quantum_circuit(
    request: QuantumExecutionRequest,
    engine: QuantumComputingEngine = Depends(get_quantum_engine)
):
    """Execute a quantum circuit"""
    try:
        start_time = time.time()
        
        # Execute quantum circuit
        results = await engine.execute_quantum_circuit(
            circuit_id=request.circuit_id,
            backend=request.backend
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "execution_results": {
                "circuit_id": request.circuit_id,
                "backend": request.backend,
                "shots": request.shots,
                "results": results,
                "timestamp": datetime.now().isoformat()
            },
            "processing_time_ms": processing_time,
            "message": "Quantum circuit executed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error executing quantum circuit: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit execution failed: {str(e)}")


@router.post("/run-algorithm", response_model=Dict[str, Any])
async def run_quantum_algorithm(
    request: QuantumAlgorithmRequest,
    engine: QuantumComputingEngine = Depends(get_quantum_engine)
):
    """Run a quantum algorithm"""
    try:
        start_time = time.time()
        
        # Run quantum algorithm
        algorithm = await engine.run_quantum_algorithm(
            algorithm_type=request.algorithm_type,
            parameters=request.parameters
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "algorithm_results": {
                "algorithm_id": algorithm.algorithm_id,
                "timestamp": algorithm.timestamp.isoformat(),
                "algorithm_type": algorithm.algorithm_type,
                "problem_size": algorithm.problem_size,
                "parameters": algorithm.parameters,
                "execution_time_ms": algorithm.execution_time,
                "success_probability": algorithm.success_probability,
                "results": algorithm.results,
                "backend_used": algorithm.backend_used
            },
            "processing_time_ms": processing_time,
            "message": f"Quantum algorithm {request.algorithm_type} executed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error running quantum algorithm: {e}")
        raise HTTPException(status_code=500, detail=f"Algorithm execution failed: {str(e)}")


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_quantum_capabilities(
    engine: QuantumComputingEngine = Depends(get_quantum_engine)
):
    """Get quantum computing capabilities"""
    try:
        # Get quantum capabilities
        capabilities = await engine.get_quantum_capabilities()
        
        return {
            "success": True,
            "quantum_capabilities": capabilities,
            "message": "Quantum capabilities retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting quantum capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quantum capabilities: {str(e)}")


@router.get("/performance-metrics", response_model=Dict[str, Any])
async def get_quantum_performance_metrics(
    engine: QuantumComputingEngine = Depends(get_quantum_engine)
):
    """Get quantum computing performance metrics"""
    try:
        # Get performance metrics
        metrics = await engine.get_quantum_performance_metrics()
        
        return {
            "success": True,
            "performance_metrics": metrics,
            "message": "Quantum performance metrics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting quantum performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quantum performance metrics: {str(e)}")


@router.post("/configure", response_model=Dict[str, Any])
async def configure_quantum_computing(
    request: QuantumConfigRequest,
    engine: QuantumComputingEngine = Depends(get_quantum_engine)
):
    """Configure quantum computing settings"""
    try:
        # Update configuration
        config = QuantumConfig(
            enable_qiskit=request.enable_qiskit,
            enable_cirq=request.enable_cirq,
            enable_pennylane=request.enable_pennylane,
            enable_ibm_quantum=request.enable_ibm_quantum,
            enable_google_quantum=request.enable_google_quantum,
            enable_ionq=request.enable_ionq,
            enable_rigetti=request.enable_rigetti,
            max_qubits=request.max_qubits,
            max_depth=request.max_depth,
            optimization_level=request.optimization_level,
            enable_error_mitigation=request.enable_error_mitigation,
            enable_quantum_ml=request.enable_quantum_ml,
            enable_quantum_optimization=request.enable_quantum_optimization,
            enable_quantum_cryptography=request.enable_quantum_cryptography,
            enable_quantum_simulation=request.enable_quantum_simulation,
            backend_type=request.backend_type,
            shots=request.shots
        )
        
        # Update engine configuration
        engine.config = config
        
        return {
            "success": True,
            "configuration": {
                "enable_qiskit": config.enable_qiskit,
                "enable_cirq": config.enable_cirq,
                "enable_pennylane": config.enable_pennylane,
                "enable_ibm_quantum": config.enable_ibm_quantum,
                "enable_google_quantum": config.enable_google_quantum,
                "enable_ionq": config.enable_ionq,
                "enable_rigetti": config.enable_rigetti,
                "max_qubits": config.max_qubits,
                "max_depth": config.max_depth,
                "optimization_level": config.optimization_level,
                "enable_error_mitigation": config.enable_error_mitigation,
                "enable_quantum_ml": config.enable_quantum_ml,
                "enable_quantum_optimization": config.enable_quantum_optimization,
                "enable_quantum_cryptography": config.enable_quantum_cryptography,
                "enable_quantum_simulation": config.enable_quantum_simulation,
                "backend_type": config.backend_type,
                "shots": config.shots
            },
            "message": "Quantum computing configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error configuring quantum computing: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/algorithms", response_model=Dict[str, Any])
async def get_available_algorithms(
    engine: QuantumComputingEngine = Depends(get_quantum_engine)
):
    """Get available quantum algorithms"""
    try:
        algorithms = {
            "search_algorithms": {
                "grover_search": {
                    "description": "Grover's search algorithm for unstructured search",
                    "complexity": "O(√N)",
                    "parameters": ["search_space", "target"],
                    "use_cases": ["database search", "optimization", "cryptanalysis"]
                }
            },
            "transform_algorithms": {
                "quantum_fourier_transform": {
                    "description": "Quantum Fourier Transform for frequency analysis",
                    "complexity": "O(n²)",
                    "parameters": ["input_state"],
                    "use_cases": ["signal processing", "Shor's algorithm", "quantum phase estimation"]
                }
            },
            "optimization_algorithms": {
                "variational_quantum_eigensolver": {
                    "description": "VQE for finding ground state energies",
                    "complexity": "O(poly(n))",
                    "parameters": ["hamiltonian", "ansatz", "optimizer"],
                    "use_cases": ["chemistry", "materials science", "optimization"]
                },
                "quantum_approximate_optimization_algorithm": {
                    "description": "QAOA for combinatorial optimization",
                    "complexity": "O(poly(n))",
                    "parameters": ["cost_hamiltonian", "mixer_hamiltonian", "p_layers"],
                    "use_cases": ["max-cut", "traveling salesman", "graph coloring"]
                }
            },
            "machine_learning_algorithms": {
                "quantum_neural_networks": {
                    "description": "Quantum neural networks for pattern recognition",
                    "complexity": "O(poly(n))",
                    "parameters": ["data", "layers", "optimizer"],
                    "use_cases": ["classification", "regression", "feature learning"]
                },
                "quantum_kernel_methods": {
                    "description": "Quantum kernel methods for SVM",
                    "complexity": "O(poly(n))",
                    "parameters": ["data", "kernel", "regularization"],
                    "use_cases": ["classification", "anomaly detection", "clustering"]
                }
            },
            "cryptography_algorithms": {
                "shor_algorithm": {
                    "description": "Shor's algorithm for integer factorization",
                    "complexity": "O((log N)³)",
                    "parameters": ["number_to_factor"],
                    "use_cases": ["cryptanalysis", "RSA breaking", "quantum supremacy"]
                },
                "quantum_key_distribution": {
                    "description": "BB84 protocol for secure key distribution",
                    "complexity": "O(n)",
                    "parameters": ["key_length", "error_rate"],
                    "use_cases": ["secure communication", "quantum cryptography", "key exchange"]
                }
            }
        }
        
        return {
            "success": True,
            "available_algorithms": algorithms,
            "message": "Available quantum algorithms retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting available algorithms: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available algorithms: {str(e)}")


@router.get("/backends", response_model=Dict[str, Any])
async def get_available_backends(
    engine: QuantumComputingEngine = Depends(get_quantum_engine)
):
    """Get available quantum backends"""
    try:
        backends = {
            "simulators": {
                "qiskit_simulator": {
                    "description": "Qiskit Aer simulator for quantum circuit simulation",
                    "max_qubits": 32,
                    "features": ["statevector", "density_matrix", "stabilizer"],
                    "status": "available"
                },
                "cirq_simulator": {
                    "description": "Cirq simulator for quantum circuit simulation",
                    "max_qubits": 30,
                    "features": ["wavefunction", "density_matrix", "stabilizer"],
                    "status": "available"
                },
                "pennylane_simulator": {
                    "description": "PennyLane simulator for quantum machine learning",
                    "max_qubits": 25,
                    "features": ["gradients", "optimization", "machine_learning"],
                    "status": "available"
                }
            },
            "hardware_backends": {
                "ibm_quantum": {
                    "description": "IBM Quantum hardware backends",
                    "max_qubits": 127,
                    "features": ["real_quantum_hardware", "error_mitigation", "calibration"],
                    "status": "requires_account",
                    "note": "Requires IBM Quantum account and API key"
                },
                "google_quantum": {
                    "description": "Google Quantum AI hardware backends",
                    "max_qubits": 70,
                    "features": ["sycamore_processor", "quantum_supremacy", "error_correction"],
                    "status": "requires_account",
                    "note": "Requires Google Quantum AI access"
                },
                "ionq": {
                    "description": "IonQ trapped ion quantum computers",
                    "max_qubits": 64,
                    "features": ["high_fidelity", "all_to_all_connectivity", "long_coherence"],
                    "status": "requires_account",
                    "note": "Requires IonQ cloud access"
                },
                "rigetti": {
                    "description": "Rigetti superconducting quantum computers",
                    "max_qubits": 80,
                    "features": ["parametric_gates", "quantum_volume", "hybrid_computing"],
                    "status": "requires_account",
                    "note": "Requires Rigetti cloud access"
                }
            }
        }
        
        return {
            "success": True,
            "available_backends": backends,
            "message": "Available quantum backends retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting available backends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available backends: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    engine: QuantumComputingEngine = Depends(get_quantum_engine)
):
    """Quantum Computing Engine health check"""
    try:
        # Check engine components
        components_status = {
            "qiskit_engine": engine.qiskit_engine is not None,
            "cirq_engine": engine.cirq_engine is not None,
            "pennylane_engine": engine.pennylane_engine is not None
        }
        
        # Get capabilities
        capabilities = await engine.get_quantum_capabilities()
        
        # Get performance metrics
        metrics = await engine.get_quantum_performance_metrics()
        
        # Determine overall health
        all_healthy = any(components_status.values())
        
        overall_health = "healthy" if all_healthy else "degraded"
        
        return {
            "status": overall_health,
            "timestamp": datetime.now().isoformat(),
            "components": components_status,
            "capabilities": capabilities,
            "performance_metrics": metrics,
            "message": "Quantum Computing Engine is operational" if overall_health == "healthy" else "Some quantum backends may not be available"
        }
        
    except Exception as e:
        logger.error(f"Error in Quantum Computing health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Quantum Computing Engine health check failed"
        }

















