"""
Machine Learning API - Advanced Implementation
============================================

Advanced machine learning API with model training, prediction, and optimization.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import datetime

from ..services import ml_service, ModelType, ModelStatus, MLAlgorithm

# Create router
router = APIRouter()

logger = logging.getLogger(__name__)


# Request/Response models
class ModelCreateRequest(BaseModel):
    """Model create request model"""
    name: str
    model_type: str
    algorithm: str
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None


class ModelTrainRequest(BaseModel):
    """Model train request model"""
    training_data: List[Dict[str, Any]]
    target_column: str
    test_size: float = 0.2
    random_state: int = 42


class ModelPredictRequest(BaseModel):
    """Model predict request model"""
    input_data: List[Dict[str, Any]]
    return_probabilities: bool = False


class ModelResponse(BaseModel):
    """Model response model"""
    model_id: str
    name: str
    type: str
    algorithm: str
    status: str
    created_at: str
    message: str


class ModelInfoResponse(BaseModel):
    """Model info response model"""
    id: str
    name: str
    type: str
    algorithm: str
    description: str
    parameters: Dict[str, Any]
    status: str
    created_at: str
    trained_at: Optional[str]
    deployed_at: Optional[str]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    mse: Optional[float]
    r2_score: Optional[float]
    training_data_size: int
    features: List[str]
    target: Optional[str]


class PredictionResponse(BaseModel):
    """Prediction response model"""
    model_id: str
    predictions: List[float]
    input_size: int
    probabilities: Optional[List[List[float]]] = None
    timestamp: str


class ModelPerformanceResponse(BaseModel):
    """Model performance response model"""
    model_id: str
    name: str
    type: str
    algorithm: str
    status: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    mse: Optional[float]
    r2_score: Optional[float]
    training_data_size: int
    features: List[str]
    target: Optional[str]


class MLStatsResponse(BaseModel):
    """ML statistics response model"""
    total_models: int
    trained_models: int
    deployed_models: int
    failed_models: int
    total_predictions: int
    models_by_type: Dict[str, int]
    models_by_algorithm: Dict[str, int]
    cached_predictions: int


# Model creation endpoints
@router.post("/models", response_model=ModelResponse)
async def create_model(request: ModelCreateRequest):
    """Create a new ML model"""
    try:
        # Validate model type
        try:
            model_type = ModelType(request.model_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type: {request.model_type}"
            )
        
        # Validate algorithm
        try:
            algorithm = MLAlgorithm(request.algorithm)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid algorithm: {request.algorithm}"
            )
        
        model_id = await ml_service.create_model(
            name=request.name,
            model_type=model_type,
            algorithm=algorithm,
            description=request.description,
            parameters=request.parameters
        )
        
        return ModelResponse(
            model_id=model_id,
            name=request.name,
            type=request.model_type,
            algorithm=request.algorithm,
            status="training",
            created_at=datetime.utcnow().isoformat(),
            message="ML model created successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create ML model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create ML model: {str(e)}"
        )


@router.post("/models/{model_id}/train")
async def train_model(model_id: str, request: ModelTrainRequest):
    """Train ML model with provided data"""
    try:
        result = await ml_service.train_model(
            model_id=model_id,
            training_data=request.training_data,
            target_column=request.target_column,
            test_size=request.test_size,
            random_state=request.random_state
        )
        
        return {
            "model_id": model_id,
            "status": result["status"],
            "metrics": result["metrics"],
            "training_data_size": result["training_data_size"],
            "message": "Model trained successfully" if result["status"] == "trained" else "Model training failed",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to train ML model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to train ML model: {str(e)}"
        )


@router.post("/models/{model_id}/predict", response_model=PredictionResponse)
async def predict(model_id: str, request: ModelPredictRequest):
    """Make predictions using trained model"""
    try:
        result = await ml_service.predict(
            model_id=model_id,
            input_data=request.input_data,
            return_probabilities=request.return_probabilities
        )
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Failed to make predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to make predictions: {str(e)}"
        )


@router.post("/models/{model_id}/deploy")
async def deploy_model(model_id: str):
    """Deploy model for production use"""
    try:
        success = await ml_service.deploy_model(model_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to deploy model"
            )
        
        return {
            "model_id": model_id,
            "status": "deployed",
            "message": "Model deployed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy model: {str(e)}"
        )


# Model management endpoints
@router.get("/models/{model_id}", response_model=ModelInfoResponse)
async def get_model(model_id: str):
    """Get model information"""
    try:
        model = await ml_service.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        return ModelInfoResponse(**model)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model: {str(e)}"
        )


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    algorithm: Optional[str] = None,
    limit: int = 100
):
    """List models with filtering"""
    try:
        # Convert string parameters to enums
        type_enum = ModelType(model_type) if model_type else None
        status_enum = ModelStatus(status) if status else None
        algorithm_enum = MLAlgorithm(algorithm) if algorithm else None
        
        models = await ml_service.list_models(
            model_type=type_enum,
            status=status_enum,
            algorithm=algorithm_enum,
            limit=limit
        )
        
        return models
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/models/{model_id}/performance", response_model=ModelPerformanceResponse)
async def get_model_performance(model_id: str):
    """Get model performance metrics"""
    try:
        performance = await ml_service.get_model_performance(model_id)
        if not performance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        return ModelPerformanceResponse(**performance)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model performance: {str(e)}"
        )


@router.post("/models/{model_id}/retrain")
async def retrain_model(
    model_id: str,
    request: ModelTrainRequest
):
    """Retrain existing model with new data"""
    try:
        result = await ml_service.retrain_model(
            model_id=model_id,
            new_training_data=request.training_data,
            target_column=request.target_column,
            test_size=request.test_size
        )
        
        return {
            "model_id": model_id,
            "status": result["status"],
            "metrics": result["metrics"],
            "training_data_size": result["training_data_size"],
            "message": "Model retrained successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to retrain model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrain model: {str(e)}"
        )


# Statistics endpoint
@router.get("/stats", response_model=MLStatsResponse)
async def get_ml_stats():
    """Get ML service statistics"""
    try:
        stats = await ml_service.get_ml_stats()
        return MLStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get ML stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ML stats: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def ml_health():
    """ML service health check"""
    try:
        stats = await ml_service.get_ml_stats()
        
        return {
            "service": "ml_service",
            "status": "healthy",
            "total_models": stats["total_models"],
            "trained_models": stats["trained_models"],
            "deployed_models": stats["deployed_models"],
            "total_predictions": stats["total_predictions"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"ML service health check failed: {e}")
        return {
            "service": "ml_service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

