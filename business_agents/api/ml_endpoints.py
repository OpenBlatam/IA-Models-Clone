"""
ML Pipeline API Endpoints
=========================

REST API endpoints for machine learning pipeline management and predictions.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd

from ..services.ml_pipeline_service import (
    MLPipelineService,
    ModelType,
    DataType,
    DataSchema,
    ModelMetrics,
    PredictionResult
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Pydantic models
class CreatePipelineRequest(BaseModel):
    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    name: str = Field(..., description="Pipeline name")
    model_type: ModelType = Field(..., description="Type of ML model")
    feature_columns: List[str] = Field(..., description="List of feature column names")
    target_column: str = Field("", description="Target column name (empty for clustering)")
    model_params: Optional[Dict[str, Any]] = Field(None, description="Model parameters")

class TrainingDataRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Training data")
    validation_split: Optional[float] = Field(0.2, description="Validation data split ratio")

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Data for prediction")

class DataAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Data to analyze")

class RetrainRequest(BaseModel):
    new_data: List[Dict[str, Any]] = Field(..., description="New training data")
    incremental: bool = Field(True, description="Whether to use incremental learning")

class PredictionExplanationRequest(BaseModel):
    prediction_data: Dict[str, Any] = Field(..., description="Single data point for explanation")

# Global ML service instance
ml_service = None

def get_ml_service() -> MLPipelineService:
    """Get global ML service instance."""
    global ml_service
    if ml_service is None:
        ml_service = MLPipelineService({"models_dir": "./models"})
    return ml_service

# API Endpoints

@router.post("/pipelines", response_model=Dict[str, str])
async def create_pipeline(
    request: CreatePipelineRequest,
    current_user: User = Depends(require_permission("ml:create"))
):
    """Create a new ML pipeline."""
    
    ml_service = get_ml_service()
    
    try:
        pipeline = await ml_service.create_pipeline(
            pipeline_id=request.pipeline_id,
            name=request.name,
            model_type=request.model_type,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            model_params=request.model_params
        )
        
        return {
            "message": f"Pipeline {request.pipeline_id} created successfully",
            "pipeline_id": pipeline.pipeline_id,
            "model_type": pipeline.model_type.value
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create pipeline: {str(e)}")

@router.get("/pipelines", response_model=List[Dict[str, Any]])
async def list_pipelines(
    current_user: User = Depends(require_permission("ml:view"))
):
    """List all available ML pipelines."""
    
    ml_service = get_ml_service()
    
    try:
        pipelines = await ml_service.list_pipelines()
        return pipelines
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {str(e)}")

@router.get("/pipelines/{pipeline_id}", response_model=Dict[str, Any])
async def get_pipeline(
    pipeline_id: str,
    current_user: User = Depends(require_permission("ml:view"))
):
    """Get pipeline details and insights."""
    
    ml_service = get_ml_service()
    
    try:
        insights = await ml_service.get_model_insights(pipeline_id)
        return insights
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline: {str(e)}")

@router.post("/pipelines/{pipeline_id}/train", response_model=Dict[str, Any])
async def train_pipeline(
    pipeline_id: str,
    request: TrainingDataRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("ml:train"))
):
    """Train ML pipeline with provided data."""
    
    ml_service = get_ml_service()
    
    try:
        # Convert data to DataFrame
        training_data = pd.DataFrame(request.data)
        
        # Split data if validation_split is specified
        validation_data = None
        if request.validation_split > 0:
            train_data, val_data = train_test_split(
                training_data, 
                test_size=request.validation_split, 
                random_state=42
            )
            training_data = train_data
            validation_data = val_data
        
        # Train pipeline
        metrics = await ml_service.train_pipeline(pipeline_id, training_data, validation_data)
        
        return {
            "message": f"Pipeline {pipeline_id} trained successfully",
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "r2_score": metrics.r2_score,
                "silhouette_score": metrics.silhouette_score,
                "cross_val_score": metrics.cross_val_score
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train pipeline: {str(e)}")

@router.post("/pipelines/{pipeline_id}/predict", response_model=Dict[str, Any])
async def predict(
    pipeline_id: str,
    request: PredictionRequest,
    current_user: User = Depends(require_permission("ml:predict"))
):
    """Make predictions using trained pipeline."""
    
    ml_service = get_ml_service()
    
    try:
        result = await ml_service.predict(pipeline_id, request.data)
        
        return {
            "predictions": result.predictions,
            "probabilities": result.probabilities,
            "confidence_scores": result.confidence_scores,
            "feature_importance": result.feature_importance
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to make predictions: {str(e)}")

@router.post("/pipelines/{pipeline_id}/explain", response_model=Dict[str, Any])
async def explain_prediction(
    pipeline_id: str,
    request: PredictionExplanationRequest,
    current_user: User = Depends(require_permission("ml:explain"))
):
    """Get explanation for a specific prediction."""
    
    ml_service = get_ml_service()
    
    try:
        explanation = await ml_service.get_prediction_explanation(
            pipeline_id, 
            request.prediction_data
        )
        
        return explanation
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to explain prediction: {str(e)}")

@router.post("/pipelines/{pipeline_id}/retrain", response_model=Dict[str, Any])
async def retrain_pipeline(
    pipeline_id: str,
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("ml:train"))
):
    """Retrain pipeline with new data."""
    
    ml_service = get_ml_service()
    
    try:
        # Convert data to DataFrame
        new_data = pd.DataFrame(request.new_data)
        
        # Retrain pipeline
        metrics = await ml_service.retrain_pipeline(
            pipeline_id, 
            new_data, 
            request.incremental
        )
        
        return {
            "message": f"Pipeline {pipeline_id} retrained successfully",
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "r2_score": metrics.r2_score,
                "silhouette_score": metrics.silhouette_score,
                "cross_val_score": metrics.cross_val_score
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrain pipeline: {str(e)}")

@router.get("/pipelines/{pipeline_id}/feature-importance", response_model=Dict[str, float])
async def get_feature_importance(
    pipeline_id: str,
    current_user: User = Depends(require_permission("ml:view"))
):
    """Get feature importance for trained model."""
    
    ml_service = get_ml_service()
    
    try:
        importance = await ml_service.get_feature_importance(pipeline_id)
        return importance
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

@router.post("/analyze-data", response_model=List[Dict[str, Any]])
async def analyze_data(
    request: DataAnalysisRequest,
    current_user: User = Depends(require_permission("ml:analyze"))
):
    """Analyze data and create schema."""
    
    ml_service = get_ml_service()
    
    try:
        # Convert data to DataFrame
        data = pd.DataFrame(request.data)
        
        # Analyze data
        schema = await ml_service.analyze_data(data)
        
        return [
            {
                "column_name": s.column_name,
                "data_type": s.data_type.value,
                "is_target": s.is_target,
                "is_feature": s.is_feature,
                "missing_percentage": s.missing_percentage,
                "unique_values": s.unique_values,
                "description": s.description
            }
            for s in schema
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze data: {str(e)}")

@router.delete("/pipelines/{pipeline_id}", response_model=Dict[str, str])
async def delete_pipeline(
    pipeline_id: str,
    current_user: User = Depends(require_permission("ml:delete"))
):
    """Delete ML pipeline."""
    
    ml_service = get_ml_service()
    
    try:
        success = await ml_service.delete_pipeline(pipeline_id)
        
        if success:
            return {"message": f"Pipeline {pipeline_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete pipeline")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete pipeline: {str(e)}")

@router.get("/model-types", response_model=List[Dict[str, str]])
async def get_model_types():
    """Get available model types."""
    
    model_types = [
        {
            "type": model_type.value,
            "name": model_type.value.replace("_", " ").title(),
            "description": get_model_description(model_type)
        }
        for model_type in ModelType
    ]
    
    return model_types

def get_model_description(model_type: ModelType) -> str:
    """Get description for model type."""
    
    descriptions = {
        ModelType.CLASSIFICATION: "Predict categorical outcomes (e.g., churn, fraud)",
        ModelType.REGRESSION: "Predict continuous values (e.g., sales, prices)",
        ModelType.CLUSTERING: "Group similar data points (e.g., customer segmentation)",
        ModelType.ANOMALY_DETECTION: "Identify unusual patterns or outliers",
        ModelType.TIME_SERIES: "Analyze time-based data patterns",
        ModelType.NLP: "Process and analyze text data",
        ModelType.RECOMMENDATION: "Recommend items to users"
    }
    
    return descriptions.get(model_type, "Machine learning model")

@router.get("/data-types", response_model=List[Dict[str, str]])
async def get_data_types():
    """Get available data types."""
    
    data_types = [
        {
            "type": data_type.value,
            "name": data_type.value.replace("_", " ").title(),
            "description": get_data_type_description(data_type)
        }
        for data_type in DataType
    ]
    
    return data_types

def get_data_type_description(data_type: DataType) -> str:
    """Get description for data type."""
    
    descriptions = {
        DataType.NUMERICAL: "Numeric values (integers, floats)",
        DataType.CATEGORICAL: "Categorical values (strings, categories)",
        DataType.TEXT: "Text data for NLP processing",
        DataType.DATETIME: "Date and time values",
        DataType.BOOLEAN: "True/false values"
    }
    
    return descriptions.get(data_type, "Data type")

@router.get("/pipelines/{pipeline_id}/health", response_model=Dict[str, Any])
async def get_pipeline_health(
    pipeline_id: str,
    current_user: User = Depends(require_permission("ml:view"))
):
    """Get pipeline health status."""
    
    ml_service = get_ml_service()
    
    try:
        insights = await ml_service.get_model_insights(pipeline_id)
        
        # Determine health status
        if not insights["is_trained"]:
            status = "not_trained"
            message = "Pipeline is not trained"
        elif insights["metrics"] and insights["metrics"]["accuracy"]:
            accuracy = insights["metrics"]["accuracy"]
            if accuracy > 0.8:
                status = "healthy"
                message = "Pipeline is performing well"
            elif accuracy > 0.6:
                status = "degraded"
                message = "Pipeline performance is acceptable"
            else:
                status = "unhealthy"
                message = "Pipeline performance is poor"
        else:
            status = "unknown"
            message = "Pipeline health cannot be determined"
        
        return {
            "pipeline_id": pipeline_id,
            "status": status,
            "message": message,
            "is_trained": insights["is_trained"],
            "last_trained": insights["last_trained"],
            "metrics": insights["metrics"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline health: {str(e)}")

@router.post("/pipelines/{pipeline_id}/save", response_model=Dict[str, str])
async def save_pipeline(
    pipeline_id: str,
    current_user: User = Depends(require_permission("ml:manage"))
):
    """Save pipeline to disk."""
    
    ml_service = get_ml_service()
    
    try:
        if pipeline_id not in ml_service.pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        pipeline = ml_service.pipelines[pipeline_id]
        await ml_service._save_pipeline(pipeline)
        
        return {"message": f"Pipeline {pipeline_id} saved successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save pipeline: {str(e)}")

@router.post("/pipelines/{pipeline_id}/load", response_model=Dict[str, str])
async def load_pipeline(
    pipeline_id: str,
    current_user: User = Depends(require_permission("ml:manage"))
):
    """Load pipeline from disk."""
    
    ml_service = get_ml_service()
    
    try:
        await ml_service.load_pipeline(pipeline_id)
        
        return {"message": f"Pipeline {pipeline_id} loaded successfully"}
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found on disk")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load pipeline: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def ml_service_health():
    """Health check for ML service."""
    
    ml_service = get_ml_service()
    
    try:
        # Check if service is working
        pipelines = await ml_service.list_pipelines()
        
        return {
            "status": "healthy",
            "service": "ML Pipeline Service",
            "timestamp": datetime.now().isoformat(),
            "pipelines_count": len(pipelines),
            "trained_pipelines": len([p for p in pipelines if p["is_trained"]]),
            "models_dir": str(ml_service.models_dir)
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "ML Pipeline Service",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }





























