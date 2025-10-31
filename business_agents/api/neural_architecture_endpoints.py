"""
Neural Architecture API Endpoints
=================================

API endpoints for neural architecture service.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.neural_architecture_service import (
    NeuralArchitectureService,
    NeuralArchitecture,
    ArchitectureSearch,
    ArchitectureOptimization,
    ArchitectureType,
    SearchStrategy,
    OptimizationObjective
)

logger = logging.getLogger(__name__)

# Create router
neural_architecture_router = APIRouter(prefix="/neural-architecture", tags=["Neural Architecture"])

# Pydantic models for request/response
class NeuralArchitectureRequest(BaseModel):
    name: str
    architecture_type: ArchitectureType
    layers: List[Dict[str, Any]]
    connections: List[Tuple[str, str]]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    metadata: Dict[str, Any] = {}

class ArchitectureSearchRequest(BaseModel):
    name: str
    search_strategy: SearchStrategy
    objectives: List[OptimizationObjective]
    constraints: Dict[str, Any] = {}
    search_space: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class ArchitectureOptimizationRequest(BaseModel):
    name: str
    architecture_id: str
    optimization_objectives: List[OptimizationObjective]
    optimization_algorithm: str
    parameters: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class NeuralArchitectureResponse(BaseModel):
    architecture_id: str
    name: str
    architecture_type: str
    layers: List[Dict[str, Any]]
    connections: List[Tuple[str, str]]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    total_parameters: int
    flops: int
    memory_usage: int
    performance_metrics: Dict[str, float]
    created_at: datetime
    metadata: Dict[str, Any]

class ArchitectureSearchResponse(BaseModel):
    search_id: str
    name: str
    search_strategy: str
    objectives: List[str]
    constraints: Dict[str, Any]
    search_space: Dict[str, Any]
    best_architectures: List[str]
    search_progress: float
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class ArchitectureOptimizationResponse(BaseModel):
    optimization_id: str
    name: str
    architecture_id: str
    optimization_objectives: List[str]
    optimization_algorithm: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class ServiceStatusResponse(BaseModel):
    service_status: str
    total_architectures: int
    total_searches: int
    total_optimizations: int
    running_searches: int
    running_optimizations: int
    layer_templates: int
    architecture_patterns: int
    search_algorithms: int
    architecture_search_enabled: bool
    neural_architecture_search_enabled: bool
    multi_objective_optimization: bool
    automated_architecture_design: bool
    architecture_visualization: bool
    performance_prediction: bool
    timestamp: str

# Dependency to get neural architecture service
async def get_neural_architecture_service() -> NeuralArchitectureService:
    """Get neural architecture service instance."""
    # This would be injected from your dependency injection system
    # For now, we'll create a mock instance
    from ..main import get_neural_architecture_service
    return await get_neural_architecture_service()

@neural_architecture_router.post("/architectures", response_model=Dict[str, str])
async def create_neural_architecture(
    request: NeuralArchitectureRequest,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Create neural architecture."""
    try:
        # Convert layers to NeuralLayer objects
        layers = []
        for i, layer_data in enumerate(request.layers):
            from ..services.neural_architecture_service import NeuralLayer
            layer = NeuralLayer(
                layer_id=f"l{i+1}",
                layer_type=layer_data.get("type", "linear"),
                parameters=layer_data.get("parameters", {}),
                input_shape=layer_data.get("input_shape", (1,)),
                output_shape=layer_data.get("output_shape", (1,)),
                activation=layer_data.get("activation"),
                dropout=layer_data.get("dropout"),
                normalization=layer_data.get("normalization"),
                metadata=layer_data.get("metadata", {})
            )
            layers.append(layer)
        
        architecture = NeuralArchitecture(
            architecture_id="",
            name=request.name,
            architecture_type=request.architecture_type,
            layers=layers,
            connections=request.connections,
            input_shape=request.input_shape,
            output_shape=request.output_shape,
            total_parameters=0,
            flops=0,
            memory_usage=0,
            performance_metrics={},
            created_at=datetime.utcnow(),
            metadata=request.metadata
        )
        
        architecture_id = await neural_architecture_service.create_neural_architecture(architecture)
        
        return {"architecture_id": architecture_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create neural architecture: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/architectures/{architecture_id}", response_model=NeuralArchitectureResponse)
async def get_neural_architecture(
    architecture_id: str,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Get neural architecture."""
    try:
        architecture = await neural_architecture_service.get_neural_architecture(architecture_id)
        
        if not architecture:
            raise HTTPException(status_code=404, detail="Neural architecture not found")
            
        # Convert layers to dictionaries
        layers_data = []
        for layer in architecture.layers:
            layer_data = {
                "layer_id": layer.layer_id,
                "layer_type": layer.layer_type,
                "parameters": layer.parameters,
                "input_shape": layer.input_shape,
                "output_shape": layer.output_shape,
                "activation": layer.activation,
                "dropout": layer.dropout,
                "normalization": layer.normalization,
                "metadata": layer.metadata
            }
            layers_data.append(layer_data)
            
        return NeuralArchitectureResponse(
            architecture_id=architecture.architecture_id,
            name=architecture.name,
            architecture_type=architecture.architecture_type.value,
            layers=layers_data,
            connections=architecture.connections,
            input_shape=architecture.input_shape,
            output_shape=architecture.output_shape,
            total_parameters=architecture.total_parameters,
            flops=architecture.flops,
            memory_usage=architecture.memory_usage,
            performance_metrics=architecture.performance_metrics,
            created_at=architecture.created_at,
            metadata=architecture.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get neural architecture: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/architectures", response_model=List[NeuralArchitectureResponse])
async def list_neural_architectures(
    architecture_type: Optional[ArchitectureType] = None,
    limit: int = 100,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """List neural architectures."""
    try:
        architectures = await neural_architecture_service.list_neural_architectures(architecture_type)
        
        result = []
        for architecture in architectures[:limit]:
            # Convert layers to dictionaries
            layers_data = []
            for layer in architecture.layers:
                layer_data = {
                    "layer_id": layer.layer_id,
                    "layer_type": layer.layer_type,
                    "parameters": layer.parameters,
                    "input_shape": layer.input_shape,
                    "output_shape": layer.output_shape,
                    "activation": layer.activation,
                    "dropout": layer.dropout,
                    "normalization": layer.normalization,
                    "metadata": layer.metadata
                }
                layers_data.append(layer_data)
                
            result.append(NeuralArchitectureResponse(
                architecture_id=architecture.architecture_id,
                name=architecture.name,
                architecture_type=architecture.architecture_type.value,
                layers=layers_data,
                connections=architecture.connections,
                input_shape=architecture.input_shape,
                output_shape=architecture.output_shape,
                total_parameters=architecture.total_parameters,
                flops=architecture.flops,
                memory_usage=architecture.memory_usage,
                performance_metrics=architecture.performance_metrics,
                created_at=architecture.created_at,
                metadata=architecture.metadata
            ))
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to list neural architectures: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.post("/search", response_model=Dict[str, str])
async def start_architecture_search(
    request: ArchitectureSearchRequest,
    background_tasks: BackgroundTasks,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Start architecture search."""
    try:
        search = ArchitectureSearch(
            search_id="",
            name=request.name,
            search_strategy=request.search_strategy,
            objectives=request.objectives,
            constraints=request.constraints,
            search_space=request.search_space,
            best_architectures=[],
            search_progress=0.0,
            status="pending",
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            metadata=request.metadata
        )
        
        search_id = await neural_architecture_service.start_architecture_search(search)
        
        return {"search_id": search_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to start architecture search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/search/{search_id}", response_model=ArchitectureSearchResponse)
async def get_architecture_search(
    search_id: str,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Get architecture search."""
    try:
        search = await neural_architecture_service.get_architecture_search(search_id)
        
        if not search:
            raise HTTPException(status_code=404, detail="Architecture search not found")
            
        return ArchitectureSearchResponse(
            search_id=search.search_id,
            name=search.name,
            search_strategy=search.search_strategy.value,
            objectives=[obj.value for obj in search.objectives],
            constraints=search.constraints,
            search_space=search.search_space,
            best_architectures=search.best_architectures,
            search_progress=search.search_progress,
            status=search.status,
            created_at=search.created_at,
            started_at=search.started_at,
            completed_at=search.completed_at,
            metadata=search.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get architecture search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/search", response_model=List[ArchitectureSearchResponse])
async def list_architecture_searches(
    status: Optional[str] = None,
    limit: int = 100,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """List architecture searches."""
    try:
        searches = await neural_architecture_service.list_architecture_searches(status)
        
        return [
            ArchitectureSearchResponse(
                search_id=search.search_id,
                name=search.name,
                search_strategy=search.search_strategy.value,
                objectives=[obj.value for obj in search.objectives],
                constraints=search.constraints,
                search_space=search.search_space,
                best_architectures=search.best_architectures,
                search_progress=search.search_progress,
                status=search.status,
                created_at=search.created_at,
                started_at=search.started_at,
                completed_at=search.completed_at,
                metadata=search.metadata
            )
            for search in searches[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list architecture searches: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.post("/optimization", response_model=Dict[str, str])
async def optimize_architecture(
    request: ArchitectureOptimizationRequest,
    background_tasks: BackgroundTasks,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Optimize architecture."""
    try:
        optimization = ArchitectureOptimization(
            optimization_id="",
            name=request.name,
            architecture_id=request.architecture_id,
            optimization_objectives=request.optimization_objectives,
            optimization_algorithm=request.optimization_algorithm,
            parameters=request.parameters,
            results={},
            status="pending",
            created_at=datetime.utcnow(),
            completed_at=None,
            metadata=request.metadata
        )
        
        optimization_id = await neural_architecture_service.optimize_architecture(optimization)
        
        return {"optimization_id": optimization_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to optimize architecture: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/optimization/{optimization_id}", response_model=ArchitectureOptimizationResponse)
async def get_architecture_optimization(
    optimization_id: str,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Get architecture optimization."""
    try:
        optimization = await neural_architecture_service.get_architecture_optimization(optimization_id)
        
        if not optimization:
            raise HTTPException(status_code=404, detail="Architecture optimization not found")
            
        return ArchitectureOptimizationResponse(
            optimization_id=optimization.optimization_id,
            name=optimization.name,
            architecture_id=optimization.architecture_id,
            optimization_objectives=[obj.value for obj in optimization.optimization_objectives],
            optimization_algorithm=optimization.optimization_algorithm,
            parameters=optimization.parameters,
            results=optimization.results,
            status=optimization.status,
            created_at=optimization.created_at,
            completed_at=optimization.completed_at,
            metadata=optimization.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get architecture optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/optimization", response_model=List[ArchitectureOptimizationResponse])
async def list_architecture_optimizations(
    status: Optional[str] = None,
    limit: int = 100,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """List architecture optimizations."""
    try:
        optimizations = await neural_architecture_service.list_architecture_optimizations(status)
        
        return [
            ArchitectureOptimizationResponse(
                optimization_id=optimization.optimization_id,
                name=optimization.name,
                architecture_id=optimization.architecture_id,
                optimization_objectives=[obj.value for obj in optimization.optimization_objectives],
                optimization_algorithm=optimization.optimization_algorithm,
                parameters=optimization.parameters,
                results=optimization.results,
                status=optimization.status,
                created_at=optimization.created_at,
                completed_at=optimization.completed_at,
                metadata=optimization.metadata
            )
            for optimization in optimizations[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list architecture optimizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/status", response_model=ServiceStatusResponse)
async def get_service_status(
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Get neural architecture service status."""
    try:
        status = await neural_architecture_service.get_service_status()
        
        return ServiceStatusResponse(
            service_status=status["service_status"],
            total_architectures=status["total_architectures"],
            total_searches=status["total_searches"],
            total_optimizations=status["total_optimizations"],
            running_searches=status["running_searches"],
            running_optimizations=status["running_optimizations"],
            layer_templates=status["layer_templates"],
            architecture_patterns=status["architecture_patterns"],
            search_algorithms=status["search_algorithms"],
            architecture_search_enabled=status["architecture_search_enabled"],
            neural_architecture_search_enabled=status["neural_architecture_search_enabled"],
            multi_objective_optimization=status["multi_objective_optimization"],
            automated_architecture_design=status["automated_architecture_design"],
            architecture_visualization=status["architecture_visualization"],
            performance_prediction=status["performance_prediction"],
            timestamp=status["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get service status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/templates", response_model=Dict[str, Any])
async def get_layer_templates(
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Get available layer templates."""
    try:
        return neural_architecture_service.layer_templates
        
    except Exception as e:
        logger.error(f"Failed to get layer templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/patterns", response_model=Dict[str, Any])
async def get_architecture_patterns(
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Get available architecture patterns."""
    try:
        return neural_architecture_service.architecture_patterns
        
    except Exception as e:
        logger.error(f"Failed to get architecture patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/algorithms", response_model=Dict[str, Any])
async def get_search_algorithms(
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Get available search algorithms."""
    try:
        return neural_architecture_service.search_algorithms
        
    except Exception as e:
        logger.error(f"Failed to get search algorithms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.get("/architecture-types", response_model=List[str])
async def get_architecture_types():
    """Get available architecture types."""
    return [arch_type.value for arch_type in ArchitectureType]

@neural_architecture_router.get("/search-strategies", response_model=List[str])
async def get_search_strategies():
    """Get available search strategies."""
    return [strategy.value for strategy in SearchStrategy]

@neural_architecture_router.get("/optimization-objectives", response_model=List[str])
async def get_optimization_objectives():
    """Get available optimization objectives."""
    return [objective.value for objective in OptimizationObjective]

@neural_architecture_router.delete("/architectures/{architecture_id}")
async def delete_neural_architecture(
    architecture_id: str,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Delete neural architecture."""
    try:
        if architecture_id not in neural_architecture_service.neural_architectures:
            raise HTTPException(status_code=404, detail="Neural architecture not found")
            
        del neural_architecture_service.neural_architectures[architecture_id]
        
        return {"status": "deleted", "architecture_id": architecture_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete neural architecture: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@neural_architecture_router.delete("/search/{search_id}")
async def delete_architecture_search(
    search_id: str,
    neural_architecture_service: NeuralArchitectureService = Depends(get_neural_architecture_service)
):
    """Delete architecture search."""
    try:
        if search_id not in neural_architecture_service.architecture_searches:
            raise HTTPException(status_code=404, detail="Architecture search not found")
            
        del neural_architecture_service.architecture_searches[search_id]
        
        return {"status": "deleted", "search_id": search_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete architecture search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
























