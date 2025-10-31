"""
Quantum API Endpoints
=====================

REST API endpoints for quantum computing integration,
quantum-enhanced optimization, and quantum cryptography.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.quantum_service import (
    QuantumService, QuantumAlgorithm, QuantumBackend, QuantumState,
    QuantumCircuit, QuantumJob, QuantumOptimization, QuantumKey
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/quantum", tags=["Quantum Computing"])

# Pydantic models
class CircuitCreateRequest(BaseModel):
    name: str = Field(..., description="Circuit name")
    qubits: int = Field(..., description="Number of qubits")
    gates: List[Dict[str, Any]] = Field(..., description="Quantum gates")
    measurements: List[Dict[str, Any]] = Field(..., description="Measurements")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Circuit parameters")

class JobExecuteRequest(BaseModel):
    circuit_id: str = Field(..., description="Circuit ID to execute")
    algorithm: str = Field(..., description="Quantum algorithm")
    backend: str = Field("simulator", description="Quantum backend")
    shots: int = Field(1024, description="Number of shots")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Job parameters")

class WorkflowOptimizationRequest(BaseModel):
    workflow_data: Dict[str, Any] = Field(..., description="Workflow data to optimize")
    algorithm: str = Field("qaoa", description="Quantum algorithm to use")

class KeyGenerationRequest(BaseModel):
    key_type: str = Field("quantum_random", description="Type of quantum key")
    security_level: str = Field("high", description="Security level")
    key_length: int = Field(256, description="Key length in bits")

# Global quantum service instance
quantum_service = None

def get_quantum_service() -> QuantumService:
    """Get global quantum service instance."""
    global quantum_service
    if quantum_service is None:
        quantum_service = QuantumService({"quantum_enabled": True})
    return quantum_service

# API Endpoints

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_quantum_service(
    current_user: User = Depends(require_permission("quantum:manage"))
):
    """Initialize the quantum service."""
    
    quantum_service = get_quantum_service()
    
    try:
        await quantum_service.initialize()
        return {"message": "Quantum Service initialized successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize quantum service: {str(e)}")

@router.get("/status", response_model=Dict[str, Any])
async def get_quantum_status(
    current_user: User = Depends(require_permission("quantum:view"))
):
    """Get quantum service status."""
    
    quantum_service = get_quantum_service()
    
    try:
        status = await quantum_service.get_service_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quantum status: {str(e)}")

@router.post("/circuits", response_model=Dict[str, Any])
async def create_quantum_circuit(
    request: CircuitCreateRequest,
    current_user: User = Depends(require_permission("quantum:manage"))
):
    """Create a new quantum circuit."""
    
    quantum_service = get_quantum_service()
    
    try:
        circuit = await quantum_service.create_quantum_circuit(
            name=request.name,
            qubits=request.qubits,
            gates=request.gates,
            measurements=request.measurements,
            parameters=request.parameters
        )
        
        return {
            "circuit_id": circuit.circuit_id,
            "name": circuit.name,
            "qubits": circuit.qubits,
            "gates_count": len(circuit.gates),
            "measurements_count": len(circuit.measurements),
            "created_at": circuit.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create quantum circuit: {str(e)}")

@router.get("/circuits", response_model=List[Dict[str, Any]])
async def get_quantum_circuits(
    current_user: User = Depends(require_permission("quantum:view"))
):
    """Get all quantum circuits."""
    
    quantum_service = get_quantum_service()
    
    try:
        circuits = await quantum_service.get_quantum_circuits()
        
        result = []
        for circuit in circuits:
            circuit_dict = {
                "circuit_id": circuit.circuit_id,
                "name": circuit.name,
                "qubits": circuit.qubits,
                "gates_count": len(circuit.gates),
                "measurements_count": len(circuit.measurements),
                "parameters": circuit.parameters,
                "created_at": circuit.created_at.isoformat(),
                "metadata": circuit.metadata
            }
            result.append(circuit_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quantum circuits: {str(e)}")

@router.get("/circuits/{circuit_id}", response_model=Dict[str, Any])
async def get_quantum_circuit(
    circuit_id: str,
    current_user: User = Depends(require_permission("quantum:view"))
):
    """Get specific quantum circuit."""
    
    quantum_service = get_quantum_service()
    
    try:
        circuits = await quantum_service.get_quantum_circuits()
        circuit = next((c for c in circuits if c.circuit_id == circuit_id), None)
        
        if not circuit:
            raise HTTPException(status_code=404, detail="Quantum circuit not found")
        
        return {
            "circuit_id": circuit.circuit_id,
            "name": circuit.name,
            "qubits": circuit.qubits,
            "gates": circuit.gates,
            "measurements": circuit.measurements,
            "parameters": circuit.parameters,
            "created_at": circuit.created_at.isoformat(),
            "metadata": circuit.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quantum circuit: {str(e)}")

@router.post("/jobs/execute", response_model=Dict[str, Any])
async def execute_quantum_job(
    request: JobExecuteRequest,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(require_permission("quantum:execute"))
):
    """Execute a quantum job."""
    
    quantum_service = get_quantum_service()
    
    try:
        # Convert string to enum
        algorithm = QuantumAlgorithm(request.algorithm)
        backend = QuantumBackend(request.backend)
        
        if background_tasks:
            # Execute in background
            background_tasks.add_task(
                quantum_service.execute_quantum_job,
                request.circuit_id,
                algorithm,
                backend,
                request.shots,
                request.parameters
            )
            return {
                "message": "Quantum job execution started in background",
                "circuit_id": request.circuit_id,
                "algorithm": request.algorithm,
                "backend": request.backend
            }
        else:
            # Execute synchronously
            job = await quantum_service.execute_quantum_job(
                circuit_id=request.circuit_id,
                algorithm=algorithm,
                backend=backend,
                shots=request.shots,
                parameters=request.parameters
            )
            
            return {
                "job_id": job.job_id,
                "circuit_id": job.circuit_id,
                "algorithm": job.algorithm.value,
                "backend": job.backend.value,
                "status": job.status.value,
                "shots": job.shots,
                "result": job.result,
                "execution_time": job.execution_time,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute quantum job: {str(e)}")

@router.get("/jobs", response_model=List[Dict[str, Any]])
async def get_quantum_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(100, description="Maximum number of jobs to return"),
    current_user: User = Depends(require_permission("quantum:view"))
):
    """Get quantum jobs."""
    
    quantum_service = get_quantum_service()
    
    try:
        # Convert string to enum if provided
        status_enum = QuantumState(status) if status else None
        
        jobs = await quantum_service.get_quantum_jobs(status_enum)
        
        # Limit results
        limited_jobs = jobs[-limit:] if limit else jobs
        
        result = []
        for job in limited_jobs:
            job_dict = {
                "job_id": job.job_id,
                "circuit_id": job.circuit_id,
                "algorithm": job.algorithm.value,
                "backend": job.backend.value,
                "shots": job.shots,
                "status": job.status.value,
                "result": job.result,
                "execution_time": job.execution_time,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "parameters": job.parameters,
                "metadata": job.metadata
            }
            result.append(job_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quantum jobs: {str(e)}")

@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_quantum_job(
    job_id: str,
    current_user: User = Depends(require_permission("quantum:view"))
):
    """Get specific quantum job."""
    
    quantum_service = get_quantum_service()
    
    try:
        jobs = await quantum_service.get_quantum_jobs()
        job = next((j for j in jobs if j.job_id == job_id), None)
        
        if not job:
            raise HTTPException(status_code=404, detail="Quantum job not found")
        
        return {
            "job_id": job.job_id,
            "circuit_id": job.circuit_id,
            "algorithm": job.algorithm.value,
            "backend": job.backend.value,
            "shots": job.shots,
            "status": job.status.value,
            "result": job.result,
            "execution_time": job.execution_time,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "parameters": job.parameters,
            "metadata": job.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quantum job: {str(e)}")

@router.post("/optimization/workflow", response_model=Dict[str, Any])
async def optimize_workflow_quantum(
    request: WorkflowOptimizationRequest,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(require_permission("quantum:execute"))
):
    """Optimize workflow using quantum algorithms."""
    
    quantum_service = get_quantum_service()
    
    try:
        # Convert string to enum
        algorithm = QuantumAlgorithm(request.algorithm)
        
        if background_tasks:
            # Execute in background
            background_tasks.add_task(
                quantum_service.optimize_workflow_quantum,
                request.workflow_data,
                algorithm
            )
            return {
                "message": "Quantum workflow optimization started in background",
                "algorithm": request.algorithm
            }
        else:
            # Execute synchronously
            optimization = await quantum_service.optimize_workflow_quantum(
                workflow_data=request.workflow_data,
                algorithm=algorithm
            )
            
            return {
                "optimization_id": optimization.optimization_id,
                "problem_type": optimization.problem_type,
                "algorithm": optimization.algorithm.value,
                "qubits_used": optimization.qubits_used,
                "iterations": optimization.iterations,
                "convergence": optimization.convergence,
                "solution": optimization.solution,
                "energy": optimization.energy,
                "execution_time": optimization.execution_time,
                "created_at": optimization.created_at.isoformat()
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize workflow quantum: {str(e)}")

@router.get("/optimizations", response_model=List[Dict[str, Any]])
async def get_quantum_optimizations(
    current_user: User = Depends(require_permission("quantum:view"))
):
    """Get quantum optimizations."""
    
    quantum_service = get_quantum_service()
    
    try:
        optimizations = await quantum_service.get_quantum_optimizations()
        
        result = []
        for optimization in optimizations:
            optimization_dict = {
                "optimization_id": optimization.optimization_id,
                "problem_type": optimization.problem_type,
                "algorithm": optimization.algorithm.value,
                "qubits_used": optimization.qubits_used,
                "iterations": optimization.iterations,
                "convergence": optimization.convergence,
                "solution": optimization.solution,
                "energy": optimization.energy,
                "execution_time": optimization.execution_time,
                "created_at": optimization.created_at.isoformat()
            }
            result.append(optimization_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quantum optimizations: {str(e)}")

@router.post("/keys/generate", response_model=Dict[str, Any])
async def generate_quantum_key(
    request: KeyGenerationRequest,
    current_user: User = Depends(require_permission("quantum:manage"))
):
    """Generate quantum cryptographic key."""
    
    quantum_service = get_quantum_service()
    
    try:
        quantum_key = await quantum_service.generate_quantum_key(
            key_type=request.key_type,
            security_level=request.security_level,
            key_length=request.key_length
        )
        
        return {
            "key_id": quantum_key.key_id,
            "key_type": quantum_key.key_type,
            "key_data": quantum_key.key_data,
            "qubits_used": quantum_key.qubits_used,
            "security_level": quantum_key.security_level,
            "created_at": quantum_key.created_at.isoformat(),
            "expires_at": quantum_key.expires_at.isoformat() if quantum_key.expires_at else None,
            "metadata": quantum_key.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quantum key: {str(e)}")

@router.get("/keys", response_model=List[Dict[str, Any]])
async def get_quantum_keys(
    current_user: User = Depends(require_permission("quantum:view"))
):
    """Get quantum keys."""
    
    quantum_service = get_quantum_service()
    
    try:
        keys = await quantum_service.get_quantum_keys()
        
        result = []
        for key in keys:
            key_dict = {
                "key_id": key.key_id,
                "key_type": key.key_type,
                "key_data": key.key_data,
                "qubits_used": key.qubits_used,
                "security_level": key.security_level,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "metadata": key.metadata
            }
            result.append(key_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quantum keys: {str(e)}")

@router.get("/backends", response_model=List[Dict[str, Any]])
async def get_available_backends(
    current_user: User = Depends(require_permission("quantum:view"))
):
    """Get available quantum backends."""
    
    quantum_service = get_quantum_service()
    
    try:
        backends = await quantum_service.get_available_backends()
        return backends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available backends: {str(e)}")

@router.get("/algorithms", response_model=List[Dict[str, Any]])
async def get_available_algorithms(
    current_user: User = Depends(require_permission("quantum:view"))
):
    """Get available quantum algorithms."""
    
    try:
        algorithms = [
            {
                "algorithm": "grover",
                "name": "Grover's Search Algorithm",
                "description": "Quantum search algorithm for finding items in unsorted database",
                "complexity": "O(√N)",
                "use_cases": ["search_optimization", "database_search", "cryptanalysis"]
            },
            {
                "algorithm": "qaoa",
                "name": "Quantum Approximate Optimization Algorithm",
                "description": "Hybrid quantum-classical algorithm for optimization problems",
                "complexity": "O(poly(n))",
                "use_cases": ["workflow_optimization", "resource_allocation", "scheduling"]
            },
            {
                "algorithm": "vqe",
                "name": "Variational Quantum Eigensolver",
                "description": "Quantum algorithm for finding eigenvalues of matrices",
                "complexity": "O(poly(n))",
                "use_cases": ["chemistry_simulation", "material_science", "optimization"]
            },
            {
                "algorithm": "qft",
                "name": "Quantum Fourier Transform",
                "description": "Quantum version of discrete Fourier transform",
                "complexity": "O(n log n)",
                "use_cases": ["signal_processing", "period_finding", "phase_estimation"]
            },
            {
                "algorithm": "shor",
                "name": "Shor's Algorithm",
                "description": "Quantum algorithm for integer factorization",
                "complexity": "O((log N)³)",
                "use_cases": ["cryptography", "number_theory", "security"]
            }
        ]
        
        return algorithms
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available algorithms: {str(e)}")

@router.get("/analytics", response_model=Dict[str, Any])
async def get_quantum_analytics(
    current_user: User = Depends(require_permission("quantum:view"))
):
    """Get quantum computing analytics."""
    
    quantum_service = get_quantum_service()
    
    try:
        # Get service status
        status = await quantum_service.get_service_status()
        
        # Get jobs
        jobs = await quantum_service.get_quantum_jobs()
        
        # Get optimizations
        optimizations = await quantum_service.get_quantum_optimizations()
        
        # Calculate analytics
        analytics = {
            "total_circuits": status.get("total_circuits", 0),
            "total_jobs": status.get("total_jobs", 0),
            "completed_jobs": status.get("completed_jobs", 0),
            "running_jobs": status.get("running_jobs", 0),
            "failed_jobs": status.get("failed_jobs", 0),
            "success_rate": (status.get("completed_jobs", 0) / max(status.get("total_jobs", 1), 1)) * 100,
            "total_optimizations": status.get("total_optimizations", 0),
            "total_keys": status.get("total_keys", 0),
            "available_backends": status.get("available_backends", 0),
            "algorithm_usage": {
                algorithm.value: len([j for j in jobs if j.algorithm == algorithm])
                for algorithm in QuantumAlgorithm
            },
            "backend_usage": {
                backend.value: len([j for j in jobs if j.backend == backend])
                for backend in QuantumBackend
            },
            "average_execution_time": sum(j.execution_time or 0 for j in jobs) / max(len(jobs), 1),
            "total_qubits_used": sum(o.qubits_used for o in optimizations),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quantum analytics: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def quantum_health_check():
    """Quantum service health check."""
    
    quantum_service = get_quantum_service()
    
    try:
        # Check if service is initialized
        initialized = hasattr(quantum_service, 'quantum_simulator') and quantum_service.quantum_simulator is not None
        
        # Get service status
        status = await quantum_service.get_service_status()
        
        return {
            "status": "healthy" if initialized else "initializing",
            "initialized": initialized,
            "simulator_available": status.get("simulator_available", False),
            "total_circuits": status.get("total_circuits", 0),
            "total_jobs": status.get("total_jobs", 0),
            "available_backends": status.get("available_backends", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }




























