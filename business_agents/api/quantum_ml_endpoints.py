"""
Quantum ML API Endpoints
========================

API endpoints for quantum machine learning service.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.quantum_ml_service import (
    QuantumMLService,
    QuantumCircuit,
    QuantumState,
    QuantumMLModel,
    QuantumOptimization,
    QuantumAlgorithm,
    QuantumGate,
    QuantumBackend,
    QuantumMLModel as QMLModel
)

logger = logging.getLogger(__name__)

# Create router
quantum_ml_router = APIRouter(prefix="/quantum-ml", tags=["Quantum Machine Learning"])

# Pydantic models for request/response
class QuantumCircuitRequest(BaseModel):
    name: str
    qubits: int
    gates: List[Dict[str, Any]]
    parameters: Dict[str, float] = {}
    metadata: Dict[str, Any] = {}

class QuantumMLModelRequest(BaseModel):
    name: str
    model_type: QMLModel
    algorithm: QuantumAlgorithm
    backend: QuantumBackend
    qubits: int
    layers: int
    parameters: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class QuantumOptimizationRequest(BaseModel):
    name: str
    algorithm: QuantumAlgorithm
    objective_function: str
    variables: int
    constraints: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class TrainingDataRequest(BaseModel):
    features: List[List[float]]
    labels: List[Any]
    test_size: float = 0.2
    random_state: int = 42

class QuantumCircuitResponse(BaseModel):
    circuit_id: str
    name: str
    qubits: int
    gates: List[Dict[str, Any]]
    parameters: Dict[str, float]
    depth: int
    created_at: datetime
    metadata: Dict[str, Any]

class QuantumStateResponse(BaseModel):
    state_id: str
    qubits: int
    amplitudes: List[complex]
    probabilities: List[float]
    fidelity: float
    created_at: datetime
    metadata: Dict[str, Any]

class QuantumMLModelResponse(BaseModel):
    model_id: str
    name: str
    model_type: str
    algorithm: str
    backend: str
    qubits: int
    layers: int
    parameters: Dict[str, Any]
    training_data: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: datetime
    last_trained: datetime
    metadata: Dict[str, Any]

class QuantumOptimizationResponse(BaseModel):
    optimization_id: str
    name: str
    algorithm: str
    objective_function: str
    variables: int
    constraints: Dict[str, Any]
    parameters: Dict[str, Any]
    result: Optional[Any]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class ServiceStatusResponse(BaseModel):
    service_status: str
    total_circuits: int
    total_states: int
    total_models: int
    total_optimizations: int
    active_optimizations: int
    quantum_backends: int
    quantum_algorithms: int
    simulation_enabled: bool
    real_quantum_enabled: bool
    error_correction_enabled: bool
    max_qubits: int
    timestamp: str

# Dependency to get quantum ML service
async def get_quantum_ml_service() -> QuantumMLService:
    """Get quantum ML service instance."""
    # This would be injected from your dependency injection system
    # For now, we'll create a mock instance
    from ..main import get_quantum_ml_service
    return await get_quantum_ml_service()

@quantum_ml_router.post("/circuits", response_model=Dict[str, str])
async def create_quantum_circuit(
    request: QuantumCircuitRequest,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Create a quantum circuit."""
    try:
        circuit = QuantumCircuit(
            circuit_id="",
            name=request.name,
            qubits=request.qubits,
            gates=request.gates,
            parameters=request.parameters,
            depth=len(request.gates),
            created_at=datetime.utcnow(),
            metadata=request.metadata
        )
        
        circuit_id = await quantum_ml_service.create_quantum_circuit(circuit)
        
        return {"circuit_id": circuit_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create quantum circuit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.post("/circuits/{circuit_id}/execute", response_model=QuantumStateResponse)
async def execute_quantum_circuit(
    circuit_id: str,
    backend: str = "simulator",
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Execute a quantum circuit."""
    try:
        state = await quantum_ml_service.execute_quantum_circuit(circuit_id, backend)
        
        return QuantumStateResponse(
            state_id=state.state_id,
            qubits=state.qubits,
            amplitudes=state.amplitudes,
            probabilities=state.probabilities,
            fidelity=state.fidelity,
            created_at=state.created_at,
            metadata=state.metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to execute quantum circuit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/circuits/{circuit_id}", response_model=QuantumCircuitResponse)
async def get_quantum_circuit(
    circuit_id: str,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Get a quantum circuit."""
    try:
        if circuit_id not in quantum_ml_service.quantum_circuits:
            raise HTTPException(status_code=404, detail="Quantum circuit not found")
            
        circuit = quantum_ml_service.quantum_circuits[circuit_id]
        
        return QuantumCircuitResponse(
            circuit_id=circuit.circuit_id,
            name=circuit.name,
            qubits=circuit.qubits,
            gates=circuit.gates,
            parameters=circuit.parameters,
            depth=circuit.depth,
            created_at=circuit.created_at,
            metadata=circuit.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quantum circuit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/circuits", response_model=List[QuantumCircuitResponse])
async def list_quantum_circuits(
    limit: int = 100,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """List quantum circuits."""
    try:
        circuits = list(quantum_ml_service.quantum_circuits.values())
        
        return [
            QuantumCircuitResponse(
                circuit_id=circuit.circuit_id,
                name=circuit.name,
                qubits=circuit.qubits,
                gates=circuit.gates,
                parameters=circuit.parameters,
                depth=circuit.depth,
                created_at=circuit.created_at,
                metadata=circuit.metadata
            )
            for circuit in circuits[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list quantum circuits: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.post("/models", response_model=Dict[str, str])
async def create_quantum_ml_model(
    request: QuantumMLModelRequest,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Create a quantum ML model."""
    try:
        model = QuantumMLModel(
            model_id="",
            name=request.name,
            model_type=request.model_type,
            algorithm=request.algorithm,
            backend=request.backend,
            qubits=request.qubits,
            layers=request.layers,
            parameters=request.parameters,
            training_data={},
            performance_metrics={},
            created_at=datetime.utcnow(),
            last_trained=datetime.utcnow(),
            metadata=request.metadata
        )
        
        model_id = await quantum_ml_service.create_quantum_ml_model(model)
        
        return {"model_id": model_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create quantum ML model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.post("/models/{model_id}/train", response_model=Dict[str, Any])
async def train_quantum_ml_model(
    model_id: str,
    request: TrainingDataRequest,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Train a quantum ML model."""
    try:
        training_data = {
            "features": request.features,
            "labels": request.labels,
            "test_size": request.test_size,
            "random_state": request.random_state
        }
        
        result = await quantum_ml_service.train_quantum_ml_model(model_id, training_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to train quantum ML model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/models/{model_id}", response_model=QuantumMLModelResponse)
async def get_quantum_ml_model(
    model_id: str,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Get a quantum ML model."""
    try:
        if model_id not in quantum_ml_service.quantum_models:
            raise HTTPException(status_code=404, detail="Quantum ML model not found")
            
        model = quantum_ml_service.quantum_models[model_id]
        
        return QuantumMLModelResponse(
            model_id=model.model_id,
            name=model.name,
            model_type=model.model_type.value,
            algorithm=model.algorithm.value,
            backend=model.backend.value,
            qubits=model.qubits,
            layers=model.layers,
            parameters=model.parameters,
            training_data=model.training_data,
            performance_metrics=model.performance_metrics,
            created_at=model.created_at,
            last_trained=model.last_trained,
            metadata=model.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quantum ML model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/models", response_model=List[QuantumMLModelResponse])
async def list_quantum_ml_models(
    limit: int = 100,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """List quantum ML models."""
    try:
        models = list(quantum_ml_service.quantum_models.values())
        
        return [
            QuantumMLModelResponse(
                model_id=model.model_id,
                name=model.name,
                model_type=model.model_type.value,
                algorithm=model.algorithm.value,
                backend=model.backend.value,
                qubits=model.qubits,
                layers=model.layers,
                parameters=model.parameters,
                training_data=model.training_data,
                performance_metrics=model.performance_metrics,
                created_at=model.created_at,
                last_trained=model.last_trained,
                metadata=model.metadata
            )
            for model in models[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list quantum ML models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.post("/optimization", response_model=Dict[str, str])
async def run_quantum_optimization(
    request: QuantumOptimizationRequest,
    background_tasks: BackgroundTasks,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Run quantum optimization."""
    try:
        optimization = QuantumOptimization(
            optimization_id="",
            name=request.name,
            algorithm=request.algorithm,
            objective_function=request.objective_function,
            variables=request.variables,
            constraints=request.constraints,
            parameters=request.parameters,
            result=None,
            status="pending",
            created_at=datetime.utcnow(),
            completed_at=None,
            metadata=request.metadata
        )
        
        optimization_id = await quantum_ml_service.run_quantum_optimization(optimization)
        
        return {"optimization_id": optimization_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to run quantum optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/optimization/{optimization_id}", response_model=QuantumOptimizationResponse)
async def get_quantum_optimization(
    optimization_id: str,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Get quantum optimization result."""
    try:
        if optimization_id not in quantum_ml_service.quantum_optimizations:
            raise HTTPException(status_code=404, detail="Quantum optimization not found")
            
        optimization = quantum_ml_service.quantum_optimizations[optimization_id]
        
        return QuantumOptimizationResponse(
            optimization_id=optimization.optimization_id,
            name=optimization.name,
            algorithm=optimization.algorithm.value,
            objective_function=optimization.objective_function,
            variables=optimization.variables,
            constraints=optimization.constraints,
            parameters=optimization.parameters,
            result=optimization.result,
            status=optimization.status,
            created_at=optimization.created_at,
            completed_at=optimization.completed_at,
            metadata=optimization.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quantum optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/optimization", response_model=List[QuantumOptimizationResponse])
async def list_quantum_optimizations(
    status: Optional[str] = None,
    limit: int = 100,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """List quantum optimizations."""
    try:
        optimizations = list(quantum_ml_service.quantum_optimizations.values())
        
        if status:
            optimizations = [opt for opt in optimizations if opt.status == status]
            
        return [
            QuantumOptimizationResponse(
                optimization_id=optimization.optimization_id,
                name=optimization.name,
                algorithm=optimization.algorithm.value,
                objective_function=optimization.objective_function,
                variables=optimization.variables,
                constraints=optimization.constraints,
                parameters=optimization.parameters,
                result=optimization.result,
                status=optimization.status,
                created_at=optimization.created_at,
                completed_at=optimization.completed_at,
                metadata=optimization.metadata
            )
            for optimization in optimizations[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list quantum optimizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/states/{state_id}", response_model=QuantumStateResponse)
async def get_quantum_state(
    state_id: str,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Get a quantum state."""
    try:
        if state_id not in quantum_ml_service.quantum_states:
            raise HTTPException(status_code=404, detail="Quantum state not found")
            
        state = quantum_ml_service.quantum_states[state_id]
        
        return QuantumStateResponse(
            state_id=state.state_id,
            qubits=state.qubits,
            amplitudes=state.amplitudes,
            probabilities=state.probabilities,
            fidelity=state.fidelity,
            created_at=state.created_at,
            metadata=state.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quantum state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/states", response_model=List[QuantumStateResponse])
async def list_quantum_states(
    limit: int = 100,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """List quantum states."""
    try:
        states = list(quantum_ml_service.quantum_states.values())
        
        return [
            QuantumStateResponse(
                state_id=state.state_id,
                qubits=state.qubits,
                amplitudes=state.amplitudes,
                probabilities=state.probabilities,
                fidelity=state.fidelity,
                created_at=state.created_at,
                metadata=state.metadata
            )
            for state in states[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list quantum states: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/status", response_model=ServiceStatusResponse)
async def get_service_status(
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Get quantum ML service status."""
    try:
        status = await quantum_ml_service.get_service_status()
        
        return ServiceStatusResponse(
            service_status=status["service_status"],
            total_circuits=status["total_circuits"],
            total_states=status["total_states"],
            total_models=status["total_models"],
            total_optimizations=status["total_optimizations"],
            active_optimizations=status["active_optimizations"],
            quantum_backends=status["quantum_backends"],
            quantum_algorithms=status["quantum_algorithms"],
            simulation_enabled=status["simulation_enabled"],
            real_quantum_enabled=status["real_quantum_enabled"],
            error_correction_enabled=status["error_correction_enabled"],
            max_qubits=status["max_qubits"],
            timestamp=status["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get service status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/backends", response_model=Dict[str, Any])
async def get_quantum_backends(
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Get available quantum backends."""
    try:
        return quantum_ml_service.quantum_backends
        
    except Exception as e:
        logger.error(f"Failed to get quantum backends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/algorithms", response_model=Dict[str, Any])
async def get_quantum_algorithms(
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Get available quantum algorithms."""
    try:
        return quantum_ml_service.quantum_algorithms
        
    except Exception as e:
        logger.error(f"Failed to get quantum algorithms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.get("/gates", response_model=List[str])
async def get_quantum_gates():
    """Get available quantum gates."""
    return [gate.value for gate in QuantumGate]

@quantum_ml_router.get("/model-types", response_model=List[str])
async def get_quantum_ml_model_types():
    """Get available quantum ML model types."""
    return [model.value for model in QMLModel]

@quantum_ml_router.delete("/circuits/{circuit_id}")
async def delete_quantum_circuit(
    circuit_id: str,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Delete a quantum circuit."""
    try:
        if circuit_id not in quantum_ml_service.quantum_circuits:
            raise HTTPException(status_code=404, detail="Quantum circuit not found")
            
        del quantum_ml_service.quantum_circuits[circuit_id]
        
        return {"status": "deleted", "circuit_id": circuit_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete quantum circuit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@quantum_ml_router.delete("/models/{model_id}")
async def delete_quantum_ml_model(
    model_id: str,
    quantum_ml_service: QuantumMLService = Depends(get_quantum_ml_service)
):
    """Delete a quantum ML model."""
    try:
        if model_id not in quantum_ml_service.quantum_models:
            raise HTTPException(status_code=404, detail="Quantum ML model not found")
            
        del quantum_ml_service.quantum_models[model_id]
        
        return {"status": "deleted", "model_id": model_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete quantum ML model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

























