"""
Temporal Computing API Endpoints
================================

API endpoints for temporal computing service.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.temporal_computing_service import (
    TemporalComputingService,
    TemporalData,
    TemporalAnalysis,
    TemporalPrediction,
    TemporalOptimization,
    TemporalOperation,
    TimeDimension,
    TemporalAlgorithm
)

logger = logging.getLogger(__name__)

# Create router
temporal_router = APIRouter(prefix="/temporal", tags=["Temporal Computing"])

# Pydantic models for request/response
class TemporalDataRequest(BaseModel):
    name: str
    time_series: List[Tuple[datetime, float]]
    time_dimension: TimeDimension
    temporal_properties: Dict[str, Any] = {}
    frequency: str = "hourly"
    metadata: Dict[str, Any] = {}

class TemporalAnalysisRequest(BaseModel):
    name: str
    data_id: str
    algorithm: TemporalAlgorithm
    parameters: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class TemporalPredictionRequest(BaseModel):
    name: str
    data_id: str
    algorithm: TemporalAlgorithm
    forecast_horizon: int = 100
    metadata: Dict[str, Any] = {}

class TemporalOptimizationRequest(BaseModel):
    name: str
    objective_function: str
    time_constraints: Dict[str, Any] = {}
    temporal_variables: List[str] = []
    optimization_algorithm: str = "genetic_algorithm"
    metadata: Dict[str, Any] = {}

class TemporalDataResponse(BaseModel):
    data_id: str
    name: str
    time_series: List[Tuple[datetime, float]]
    time_dimension: str
    temporal_properties: Dict[str, Any]
    frequency: str
    created_at: datetime
    metadata: Dict[str, Any]

class TemporalAnalysisResponse(BaseModel):
    analysis_id: str
    name: str
    data_id: str
    algorithm: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    accuracy: float
    confidence: float
    created_at: datetime
    completed_at: datetime
    metadata: Dict[str, Any]

class TemporalPredictionResponse(BaseModel):
    prediction_id: str
    name: str
    data_id: str
    algorithm: str
    forecast_horizon: int
    predictions: List[Tuple[datetime, float]]
    confidence_intervals: List[Tuple[float, float]]
    accuracy_metrics: Dict[str, float]
    created_at: datetime
    metadata: Dict[str, Any]

class TemporalOptimizationResponse(BaseModel):
    optimization_id: str
    name: str
    objective_function: str
    time_constraints: Dict[str, Any]
    temporal_variables: List[str]
    optimization_algorithm: str
    result: Optional[Dict[str, Any]]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class ServiceStatusResponse(BaseModel):
    service_status: str
    total_data: int
    total_analyses: int
    total_predictions: int
    total_optimizations: int
    running_optimizations: int
    temporal_models: int
    time_engines: int
    time_travel_enabled: bool
    temporal_analysis_enabled: bool
    prediction_enabled: bool
    optimization_enabled: bool
    real_time_processing: bool
    max_data_series: int
    timestamp: str

# Dependency to get temporal computing service
async def get_temporal_service() -> TemporalComputingService:
    """Get temporal computing service instance."""
    # This would be injected from your dependency injection system
    # For now, we'll create a mock instance
    from ..main import get_temporal_computing_service
    return await get_temporal_computing_service()

@temporal_router.post("/data", response_model=Dict[str, str])
async def create_temporal_data(
    request: TemporalDataRequest,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Create temporal data."""
    try:
        data = TemporalData(
            data_id="",
            name=request.name,
            time_series=request.time_series,
            time_dimension=request.time_dimension,
            temporal_properties=request.temporal_properties,
            frequency=request.frequency,
            created_at=datetime.utcnow(),
            metadata=request.metadata
        )
        
        data_id = await temporal_service.create_temporal_data(data)
        
        return {"data_id": data_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create temporal data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/data/{data_id}", response_model=TemporalDataResponse)
async def get_temporal_data(
    data_id: str,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Get temporal data."""
    try:
        data = await temporal_service.get_temporal_data(data_id)
        
        if not data:
            raise HTTPException(status_code=404, detail="Temporal data not found")
            
        return TemporalDataResponse(
            data_id=data.data_id,
            name=data.name,
            time_series=data.time_series,
            time_dimension=data.time_dimension.value,
            temporal_properties=data.temporal_properties,
            frequency=data.frequency,
            created_at=data.created_at,
            metadata=data.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get temporal data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/data", response_model=List[TemporalDataResponse])
async def list_temporal_data(
    time_dimension: Optional[TimeDimension] = None,
    limit: int = 100,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """List temporal data."""
    try:
        data_list = await temporal_service.list_temporal_data(time_dimension)
        
        return [
            TemporalDataResponse(
                data_id=data.data_id,
                name=data.name,
                time_series=data.time_series,
                time_dimension=data.time_dimension.value,
                temporal_properties=data.temporal_properties,
                frequency=data.frequency,
                created_at=data.created_at,
                metadata=data.metadata
            )
            for data in data_list[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list temporal data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.post("/analysis", response_model=Dict[str, str])
async def analyze_temporal_data(
    request: TemporalAnalysisRequest,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Analyze temporal data."""
    try:
        analysis = TemporalAnalysis(
            analysis_id="",
            name=request.name,
            data_id=request.data_id,
            algorithm=request.algorithm,
            parameters=request.parameters,
            results={},
            accuracy=0.0,
            confidence=0.0,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            metadata=request.metadata
        )
        
        analysis_id = await temporal_service.analyze_temporal_data(analysis)
        
        return {"analysis_id": analysis_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to analyze temporal data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/analysis/{analysis_id}", response_model=TemporalAnalysisResponse)
async def get_temporal_analysis(
    analysis_id: str,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Get temporal analysis."""
    try:
        analysis = await temporal_service.get_temporal_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Temporal analysis not found")
            
        return TemporalAnalysisResponse(
            analysis_id=analysis.analysis_id,
            name=analysis.name,
            data_id=analysis.data_id,
            algorithm=analysis.algorithm.value,
            parameters=analysis.parameters,
            results=analysis.results,
            accuracy=analysis.accuracy,
            confidence=analysis.confidence,
            created_at=analysis.created_at,
            completed_at=analysis.completed_at,
            metadata=analysis.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get temporal analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/analysis", response_model=List[TemporalAnalysisResponse])
async def list_temporal_analyses(
    algorithm: Optional[TemporalAlgorithm] = None,
    limit: int = 100,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """List temporal analyses."""
    try:
        analyses = await temporal_service.list_temporal_analyses(algorithm)
        
        return [
            TemporalAnalysisResponse(
                analysis_id=analysis.analysis_id,
                name=analysis.name,
                data_id=analysis.data_id,
                algorithm=analysis.algorithm.value,
                parameters=analysis.parameters,
                results=analysis.results,
                accuracy=analysis.accuracy,
                confidence=analysis.confidence,
                created_at=analysis.created_at,
                completed_at=analysis.completed_at,
                metadata=analysis.metadata
            )
            for analysis in analyses[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list temporal analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.post("/prediction", response_model=Dict[str, str])
async def predict_temporal_data(
    request: TemporalPredictionRequest,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Predict temporal data."""
    try:
        prediction = TemporalPrediction(
            prediction_id="",
            name=request.name,
            data_id=request.data_id,
            algorithm=request.algorithm,
            forecast_horizon=request.forecast_horizon,
            predictions=[],
            confidence_intervals=[],
            accuracy_metrics={},
            created_at=datetime.utcnow(),
            metadata=request.metadata
        )
        
        prediction_id = await temporal_service.predict_temporal_data(prediction)
        
        return {"prediction_id": prediction_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to predict temporal data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/prediction/{prediction_id}", response_model=TemporalPredictionResponse)
async def get_temporal_prediction(
    prediction_id: str,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Get temporal prediction."""
    try:
        prediction = await temporal_service.get_temporal_prediction(prediction_id)
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Temporal prediction not found")
            
        return TemporalPredictionResponse(
            prediction_id=prediction.prediction_id,
            name=prediction.name,
            data_id=prediction.data_id,
            algorithm=prediction.algorithm.value,
            forecast_horizon=prediction.forecast_horizon,
            predictions=prediction.predictions,
            confidence_intervals=prediction.confidence_intervals,
            accuracy_metrics=prediction.accuracy_metrics,
            created_at=prediction.created_at,
            metadata=prediction.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get temporal prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/prediction", response_model=List[TemporalPredictionResponse])
async def list_temporal_predictions(
    algorithm: Optional[TemporalAlgorithm] = None,
    limit: int = 100,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """List temporal predictions."""
    try:
        predictions = await temporal_service.list_temporal_predictions(algorithm)
        
        return [
            TemporalPredictionResponse(
                prediction_id=prediction.prediction_id,
                name=prediction.name,
                data_id=prediction.data_id,
                algorithm=prediction.algorithm.value,
                forecast_horizon=prediction.forecast_horizon,
                predictions=prediction.predictions,
                confidence_intervals=prediction.confidence_intervals,
                accuracy_metrics=prediction.accuracy_metrics,
                created_at=prediction.created_at,
                metadata=prediction.metadata
            )
            for prediction in predictions[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list temporal predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.post("/optimization", response_model=Dict[str, str])
async def optimize_temporal_system(
    request: TemporalOptimizationRequest,
    background_tasks: BackgroundTasks,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Optimize temporal system."""
    try:
        optimization = TemporalOptimization(
            optimization_id="",
            name=request.name,
            objective_function=request.objective_function,
            time_constraints=request.time_constraints,
            temporal_variables=request.temporal_variables,
            optimization_algorithm=request.optimization_algorithm,
            result=None,
            status="pending",
            created_at=datetime.utcnow(),
            completed_at=None,
            metadata=request.metadata
        )
        
        optimization_id = await temporal_service.optimize_temporal_system(optimization)
        
        return {"optimization_id": optimization_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to optimize temporal system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/optimization/{optimization_id}", response_model=TemporalOptimizationResponse)
async def get_temporal_optimization(
    optimization_id: str,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Get temporal optimization result."""
    try:
        optimization = await temporal_service.get_temporal_optimization(optimization_id)
        
        if not optimization:
            raise HTTPException(status_code=404, detail="Temporal optimization not found")
            
        return TemporalOptimizationResponse(
            optimization_id=optimization.optimization_id,
            name=optimization.name,
            objective_function=optimization.objective_function,
            time_constraints=optimization.time_constraints,
            temporal_variables=optimization.temporal_variables,
            optimization_algorithm=optimization.optimization_algorithm,
            result=optimization.result,
            status=optimization.status,
            created_at=optimization.created_at,
            completed_at=optimization.completed_at,
            metadata=optimization.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get temporal optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/optimization", response_model=List[TemporalOptimizationResponse])
async def list_temporal_optimizations(
    status: Optional[str] = None,
    limit: int = 100,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """List temporal optimizations."""
    try:
        optimizations = await temporal_service.list_temporal_optimizations(status)
        
        return [
            TemporalOptimizationResponse(
                optimization_id=optimization.optimization_id,
                name=optimization.name,
                objective_function=optimization.objective_function,
                time_constraints=optimization.time_constraints,
                temporal_variables=optimization.temporal_variables,
                optimization_algorithm=optimization.optimization_algorithm,
                result=optimization.result,
                status=optimization.status,
                created_at=optimization.created_at,
                completed_at=optimization.completed_at,
                metadata=optimization.metadata
            )
            for optimization in optimizations[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list temporal optimizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/status", response_model=ServiceStatusResponse)
async def get_service_status(
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Get temporal computing service status."""
    try:
        status = await temporal_service.get_service_status()
        
        return ServiceStatusResponse(
            service_status=status["service_status"],
            total_data=status["total_data"],
            total_analyses=status["total_analyses"],
            total_predictions=status["total_predictions"],
            total_optimizations=status["total_optimizations"],
            running_optimizations=status["running_optimizations"],
            temporal_models=status["temporal_models"],
            time_engines=status["time_engines"],
            time_travel_enabled=status["time_travel_enabled"],
            temporal_analysis_enabled=status["temporal_analysis_enabled"],
            prediction_enabled=status["prediction_enabled"],
            optimization_enabled=status["optimization_enabled"],
            real_time_processing=status["real_time_processing"],
            max_data_series=status["max_data_series"],
            timestamp=status["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get service status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/models", response_model=Dict[str, Any])
async def get_temporal_models(
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Get available temporal models."""
    try:
        return temporal_service.temporal_models
        
    except Exception as e:
        logger.error(f"Failed to get temporal models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/engines", response_model=Dict[str, Any])
async def get_time_engines(
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Get available time engines."""
    try:
        return temporal_service.time_engines
        
    except Exception as e:
        logger.error(f"Failed to get time engines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@temporal_router.get("/operations", response_model=List[str])
async def get_temporal_operations():
    """Get available temporal operations."""
    return [operation.value for operation in TemporalOperation]

@temporal_router.get("/dimensions", response_model=List[str])
async def get_time_dimensions():
    """Get available time dimensions."""
    return [dimension.value for dimension in TimeDimension]

@temporal_router.get("/algorithms", response_model=List[str])
async def get_temporal_algorithms():
    """Get available temporal algorithms."""
    return [algorithm.value for algorithm in TemporalAlgorithm]

@temporal_router.delete("/data/{data_id}")
async def delete_temporal_data(
    data_id: str,
    temporal_service: TemporalComputingService = Depends(get_temporal_service)
):
    """Delete temporal data."""
    try:
        if data_id not in temporal_service.temporal_data:
            raise HTTPException(status_code=404, detail="Temporal data not found")
            
        del temporal_service.temporal_data[data_id]
        
        return {"status": "deleted", "data_id": data_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete temporal data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
























