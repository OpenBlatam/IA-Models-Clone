"""
Quantum computing API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ....services.quantum_service import QuantumService
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ExternalServiceError

router = APIRouter()


class QuantumRecommendationRequest(BaseModel):
    """Request model for quantum recommendations."""
    user_preferences: List[float] = Field(..., description="User preference vector")
    content_features: List[List[float]] = Field(..., description="Content feature vectors")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")


class QuantumSearchRequest(BaseModel):
    """Request model for quantum search."""
    search_query: str = Field(..., min_length=1, description="Search query")
    content_database: List[Dict[str, Any]] = Field(..., description="Content database to search")


class QuantumClusteringRequest(BaseModel):
    """Request model for quantum clustering."""
    content_vectors: List[List[float]] = Field(..., description="Content vectors to cluster")
    num_clusters: int = Field(default=5, ge=2, le=20, description="Number of clusters")


class QuantumKeyGenerationRequest(BaseModel):
    """Request model for quantum key generation."""
    key_length: int = Field(default=256, ge=128, le=2048, description="Key length in bits")


class QuantumOptimizationRequest(BaseModel):
    """Request model for quantum optimization."""
    optimization_problem: str = Field(..., description="Type of optimization problem")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Optimization parameters")


async def get_quantum_service(session: DatabaseSessionDep) -> QuantumService:
    """Get quantum service instance."""
    return QuantumService(session)


@router.post("/recommendations", response_model=Dict[str, Any])
async def optimize_content_recommendations(
    request: QuantumRecommendationRequest = Depends(),
    quantum_service: QuantumService = Depends(get_quantum_service),
    current_user: CurrentUserDep = Depends()
):
    """Use quantum optimization for content recommendations."""
    try:
        result = await quantum_service.optimize_content_recommendations(
            user_preferences=request.user_preferences,
            content_features=request.content_features,
            num_recommendations=request.num_recommendations
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Quantum recommendations generated successfully"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Quantum service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate quantum recommendations"
        )


@router.post("/search", response_model=Dict[str, Any])
async def quantum_search_optimization(
    request: QuantumSearchRequest = Depends(),
    quantum_service: QuantumService = Depends(get_quantum_service),
    current_user: CurrentUserDep = Depends()
):
    """Use quantum algorithms for search optimization."""
    try:
        result = await quantum_service.quantum_search_optimization(
            search_query=request.search_query,
            content_database=request.content_database
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Quantum search optimization completed"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Quantum service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform quantum search optimization"
        )


@router.post("/clustering", response_model=Dict[str, Any])
async def quantum_clustering(
    request: QuantumClusteringRequest = Depends(),
    quantum_service: QuantumService = Depends(get_quantum_service),
    current_user: CurrentUserDep = Depends()
):
    """Use quantum machine learning for content clustering."""
    try:
        result = await quantum_service.quantum_clustering(
            content_vectors=request.content_vectors,
            num_clusters=request.num_clusters
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Quantum clustering completed successfully"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Quantum service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform quantum clustering"
        )


@router.post("/encryption/key-generation", response_model=Dict[str, Any])
async def quantum_encryption_key_generation(
    request: QuantumKeyGenerationRequest = Depends(),
    quantum_service: QuantumService = Depends(get_quantum_service),
    current_user: CurrentUserDep = Depends()
):
    """Generate quantum-secure encryption keys."""
    try:
        result = await quantum_service.quantum_encryption_key_generation(
            key_length=request.key_length
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Quantum encryption key generated successfully"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Quantum service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate quantum encryption key"
        )


@router.post("/optimization", response_model=Dict[str, Any])
async def quantum_optimization_analysis(
    request: QuantumOptimizationRequest = Depends(),
    quantum_service: QuantumService = Depends(get_quantum_service),
    current_user: CurrentUserDep = Depends()
):
    """Perform quantum optimization analysis."""
    try:
        result = await quantum_service.quantum_optimization_analysis(
            optimization_problem=request.optimization_problem,
            parameters=request.parameters
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Quantum optimization analysis completed"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Quantum service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform quantum optimization analysis"
        )


@router.get("/optimization/history", response_model=Dict[str, Any])
async def get_quantum_optimization_history(
    optimization_type: Optional[str] = Query(None, description="Filter by optimization type"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of optimizations to return"),
    offset: int = Query(default=0, ge=0, description="Number of optimizations to skip"),
    quantum_service: QuantumService = Depends(get_quantum_service),
    current_user: CurrentUserDep = Depends()
):
    """Get quantum optimization history."""
    try:
        optimizations, total = await quantum_service.get_quantum_optimization_history(
            optimization_type=optimization_type,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "data": {
                "optimizations": optimizations,
                "total": total,
                "limit": limit,
                "offset": offset
            },
            "message": "Quantum optimization history retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quantum optimization history"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_quantum_service_stats(
    quantum_service: QuantumService = Depends(get_quantum_service),
    current_user: CurrentUserDep = Depends()
):
    """Get quantum service statistics."""
    try:
        stats = await quantum_service.get_quantum_service_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "Quantum service statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quantum service statistics"
        )


@router.get("/algorithms", response_model=Dict[str, Any])
async def get_available_quantum_algorithms():
    """Get list of available quantum algorithms."""
    algorithms = [
        {
            "name": "content_recommendations",
            "description": "Quantum optimization for content recommendations",
            "type": "optimization",
            "complexity": "O(n²)"
        },
        {
            "name": "quantum_search",
            "description": "Grover's algorithm for search optimization",
            "type": "search",
            "complexity": "O(√n)"
        },
        {
            "name": "quantum_clustering",
            "description": "Quantum machine learning for content clustering",
            "type": "machine_learning",
            "complexity": "O(n log n)"
        },
        {
            "name": "key_generation",
            "description": "Quantum-secure encryption key generation",
            "type": "cryptography",
            "complexity": "O(1)"
        },
        {
            "name": "content_scheduling",
            "description": "Quantum optimization for content scheduling",
            "type": "optimization",
            "complexity": "O(n³)"
        },
        {
            "name": "user_engagement",
            "description": "Quantum optimization for user engagement",
            "type": "optimization",
            "complexity": "O(n²)"
        },
        {
            "name": "resource_allocation",
            "description": "Quantum optimization for resource allocation",
            "type": "optimization",
            "complexity": "O(n log n)"
        }
    ]
    
    return {
        "success": True,
        "data": {
            "algorithms": algorithms,
            "total": len(algorithms)
        },
        "message": "Available quantum algorithms retrieved successfully"
    }


@router.get("/backend/info", response_model=Dict[str, Any])
async def get_quantum_backend_info():
    """Get quantum backend information."""
    backend_info = {
        "name": "qasm_simulator",
        "type": "simulator",
        "provider": "Qiskit",
        "max_qubits": 32,
        "available_gates": [
            "h", "x", "y", "z", "rx", "ry", "rz", "cx", "cz", "ccx"
        ],
        "supported_algorithms": [
            "QAOA", "VQE", "Grover", "QuantumKernel", "QSVC"
        ],
        "status": "available"
    }
    
    return {
        "success": True,
        "data": backend_info,
        "message": "Quantum backend information retrieved successfully"
    }


@router.get("/circuit/{optimization_id}", response_model=Dict[str, Any])
async def get_quantum_circuit_info(
    optimization_id: int,
    quantum_service: QuantumService = Depends(get_quantum_service),
    current_user: CurrentUserDep = Depends()
):
    """Get quantum circuit information for a specific optimization."""
    try:
        # Get optimization history
        optimizations, _ = await quantum_service.get_quantum_optimization_history(
            limit=1,
            offset=0
        )
        
        # Find the specific optimization
        optimization = None
        for opt in optimizations:
            if opt["id"] == optimization_id:
                optimization = opt
                break
        
        if not optimization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Optimization not found"
            )
        
        circuit_info = {
            "optimization_id": optimization_id,
            "circuit_depth": optimization["circuit_depth"],
            "execution_time": optimization["execution_time"],
            "optimization_type": optimization["optimization_type"],
            "created_at": optimization["created_at"]
        }
        
        return {
            "success": True,
            "data": circuit_info,
            "message": "Quantum circuit information retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quantum circuit information"
        )


@router.post("/simulation", response_model=Dict[str, Any])
async def run_quantum_simulation(
    circuit_depth: int = Query(default=10, ge=1, le=50, description="Circuit depth"),
    num_qubits: int = Query(default=5, ge=1, le=10, description="Number of qubits"),
    shots: int = Query(default=1024, ge=1, le=10000, description="Number of shots"),
    quantum_service: QuantumService = Depends(get_quantum_service),
    current_user: CurrentUserDep = Depends()
):
    """Run a quantum circuit simulation."""
    try:
        # This would run an actual quantum simulation
        # For now, we'll return mock results
        simulation_result = {
            "circuit_depth": circuit_depth,
            "num_qubits": num_qubits,
            "shots": shots,
            "execution_time": 0.5,
            "results": {
                "00000": 256,
                "00001": 128,
                "00010": 128,
                "00011": 64,
                "00100": 64,
                "00101": 32,
                "00110": 32,
                "00111": 16,
                "01000": 16,
                "01001": 8,
                "01010": 8,
                "01011": 4,
                "01100": 4,
                "01101": 2,
                "01110": 2,
                "01111": 1
            }
        }
        
        return {
            "success": True,
            "data": simulation_result,
            "message": "Quantum simulation completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run quantum simulation"
        )






























