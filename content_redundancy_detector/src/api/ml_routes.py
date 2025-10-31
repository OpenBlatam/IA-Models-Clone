"""
ML Routes - Advanced machine learning API for content analysis and prediction
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from pydantic import BaseModel, Field, validator
import json

from ..core.content_ml_engine import (
    train_content_classifier,
    predict_content_class,
    train_content_clustering,
    predict_content_cluster,
    train_topic_modeling,
    predict_topics,
    get_ml_model_info,
    list_ml_models,
    get_ml_engine_metrics,
    get_ml_engine_health,
    initialize_content_ml_engine,
    MLModel,
    MLPrediction
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["Machine Learning"])


# Pydantic models for request/response validation
class TrainingDataRequest(BaseModel):
    """Request model for training data"""
    texts: List[str] = Field(..., min_items=10, max_items=10000, description="List of text samples for training")
    labels: List[str] = Field(..., min_items=10, max_items=10000, description="List of corresponding labels")
    model_name: str = Field(default="content_classifier", description="Name for the model")
    model_type: str = Field(default="neural_network", description="Type of model to train")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for text in v:
            if not text.strip():
                raise ValueError('Text samples cannot be empty')
        return v
    
    @validator('labels')
    def validate_labels(cls, v):
        if not v:
            raise ValueError('Labels list cannot be empty')
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ["neural_network", "random_forest", "gradient_boosting"]
        if v not in allowed_types:
            raise ValueError(f'Model type must be one of: {allowed_types}')
        return v


class ClusteringDataRequest(BaseModel):
    """Request model for clustering training data"""
    texts: List[str] = Field(..., min_items=10, max_items=10000, description="List of text samples for clustering")
    n_clusters: int = Field(default=5, ge=2, le=20, description="Number of clusters")
    model_name: str = Field(default="content_clustering", description="Name for the clustering model")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for text in v:
            if not text.strip():
                raise ValueError('Text samples cannot be empty')
        return v


class TopicModelingDataRequest(BaseModel):
    """Request model for topic modeling training data"""
    texts: List[str] = Field(..., min_items=10, max_items=10000, description="List of text samples for topic modeling")
    n_topics: int = Field(default=10, ge=2, le=50, description="Number of topics")
    model_name: str = Field(default="topic_modeling", description="Name for the topic modeling model")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for text in v:
            if not text.strip():
                raise ValueError('Text samples cannot be empty')
        return v


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    model_id: str = Field(..., description="ID of the trained model to use")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()


class TopicPredictionRequest(BaseModel):
    """Request model for topic predictions"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze for topics")
    model_id: str = Field(..., description="ID of the trained topic modeling model")
    top_n: int = Field(default=3, ge=1, le=10, description="Number of top topics to return")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()


# Response models
class MLModelResponse(BaseModel):
    """Response model for ML model information"""
    model_id: str
    model_type: str
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_data_size: int
    features_count: int
    created_at: str
    last_trained: str
    is_active: bool


class MLPredictionResponse(BaseModel):
    """Response model for ML predictions"""
    model_id: str
    prediction: Any
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    prediction_timestamp: str


class TrainingResultResponse(BaseModel):
    """Response model for training results"""
    model_id: str
    model_type: str
    model_name: str
    training_status: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_data_size: int
    features_count: int
    training_time: float
    training_timestamp: str


# Dependency functions
async def get_current_user() -> Dict[str, str]:
    """Dependency to get current user (placeholder for auth)"""
    return {"user_id": "anonymous", "role": "user"}


async def validate_api_key(api_key: Optional[str] = Query(None)) -> bool:
    """Dependency to validate API key"""
    # Placeholder for API key validation
    return True


# Route handlers
@router.post("/train/classifier", response_model=TrainingResultResponse)
async def train_classifier_endpoint(
    request: TrainingDataRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> TrainingResultResponse:
    """
    Train a content classification model
    
    - **texts**: List of text samples for training
    - **labels**: List of corresponding labels
    - **model_name**: Name for the model
    - **model_type**: Type of model to train (neural_network, random_forest, gradient_boosting)
    """
    
    try:
        # Validate data consistency
        if len(request.texts) != len(request.labels):
            raise ValueError("Number of texts and labels must match")
        
        # Train model
        model_id = await train_content_classifier(
            texts=request.texts,
            labels=request.labels,
            model_name=request.model_name,
            model_type=request.model_type
        )
        
        # Get model info
        model_info = await get_ml_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=500, detail="Failed to retrieve model information")
        
        # Create response
        response = TrainingResultResponse(
            model_id=model_info.model_id,
            model_type=model_info.model_type,
            model_name=model_info.model_name,
            training_status="completed",
            accuracy=model_info.accuracy,
            precision=model_info.precision,
            recall=model_info.recall,
            f1_score=model_info.f1_score,
            training_data_size=model_info.training_data_size,
            features_count=model_info.features_count,
            training_time=0.0,  # Would be calculated during training
            training_timestamp=model_info.last_trained.isoformat()
        )
        
        logger.info(f"Content classifier trained successfully: {model_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in classifier training: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error training classifier: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during classifier training")


@router.post("/train/clustering", response_model=TrainingResultResponse)
async def train_clustering_endpoint(
    request: ClusteringDataRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> TrainingResultResponse:
    """
    Train a content clustering model
    
    - **texts**: List of text samples for clustering
    - **n_clusters**: Number of clusters to create
    - **model_name**: Name for the clustering model
    """
    
    try:
        # Train clustering model
        model_id = await train_content_clustering(
            texts=request.texts,
            n_clusters=request.n_clusters,
            model_name=request.model_name
        )
        
        # Get model info
        model_info = await get_ml_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=500, detail="Failed to retrieve model information")
        
        # Create response
        response = TrainingResultResponse(
            model_id=model_info.model_id,
            model_type=model_info.model_type,
            model_name=model_info.model_name,
            training_status="completed",
            accuracy=model_info.accuracy,  # Silhouette score for clustering
            precision=0.0,  # Not applicable for clustering
            recall=0.0,     # Not applicable for clustering
            f1_score=0.0,   # Not applicable for clustering
            training_data_size=model_info.training_data_size,
            features_count=model_info.features_count,
            training_time=0.0,  # Would be calculated during training
            training_timestamp=model_info.last_trained.isoformat()
        )
        
        logger.info(f"Content clustering model trained successfully: {model_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in clustering training: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error training clustering model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during clustering training")


@router.post("/train/topic-modeling", response_model=TrainingResultResponse)
async def train_topic_modeling_endpoint(
    request: TopicModelingDataRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> TrainingResultResponse:
    """
    Train a topic modeling model using LDA
    
    - **texts**: List of text samples for topic modeling
    - **n_topics**: Number of topics to discover
    - **model_name**: Name for the topic modeling model
    """
    
    try:
        # Train topic modeling
        model_id = await train_topic_modeling(
            texts=request.texts,
            n_topics=request.n_topics,
            model_name=request.model_name
        )
        
        # Get model info
        model_info = await get_ml_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=500, detail="Failed to retrieve model information")
        
        # Create response
        response = TrainingResultResponse(
            model_id=model_info.model_id,
            model_type=model_info.model_type,
            model_name=model_info.model_name,
            training_status="completed",
            accuracy=0.0,  # Not applicable for topic modeling
            precision=0.0,  # Not applicable for topic modeling
            recall=0.0,     # Not applicable for topic modeling
            f1_score=0.0,   # Not applicable for topic modeling
            training_data_size=model_info.training_data_size,
            features_count=model_info.features_count,
            training_time=0.0,  # Would be calculated during training
            training_timestamp=model_info.last_trained.isoformat()
        )
        
        logger.info(f"Topic modeling trained successfully: {model_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in topic modeling training: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error training topic modeling: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during topic modeling training")


@router.post("/predict/class", response_model=MLPredictionResponse)
async def predict_class_endpoint(
    request: PredictionRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> MLPredictionResponse:
    """
    Predict content class using a trained classification model
    
    - **text**: Text to analyze
    - **model_id**: ID of the trained model to use
    """
    
    try:
        # Make prediction
        prediction = await predict_content_class(
            text=request.text,
            model_id=request.model_id
        )
        
        # Create response
        response = MLPredictionResponse(
            model_id=prediction.model_id,
            prediction=prediction.prediction,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            feature_importance=prediction.feature_importance,
            prediction_timestamp=prediction.prediction_timestamp.isoformat()
        )
        
        logger.info(f"Content class prediction completed for model: {request.model_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in class prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting content class: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during class prediction")


@router.post("/predict/cluster", response_model=MLPredictionResponse)
async def predict_cluster_endpoint(
    request: PredictionRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> MLPredictionResponse:
    """
    Predict content cluster using a trained clustering model
    
    - **text**: Text to analyze
    - **model_id**: ID of the trained clustering model to use
    """
    
    try:
        # Make prediction
        prediction = await predict_content_cluster(
            text=request.text,
            model_id=request.model_id
        )
        
        # Create response
        response = MLPredictionResponse(
            model_id=prediction.model_id,
            prediction=prediction.prediction,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            feature_importance=prediction.feature_importance,
            prediction_timestamp=prediction.prediction_timestamp.isoformat()
        )
        
        logger.info(f"Content cluster prediction completed for model: {request.model_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in cluster prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting content cluster: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during cluster prediction")


@router.post("/predict/topics", response_model=MLPredictionResponse)
async def predict_topics_endpoint(
    request: TopicPredictionRequest,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> MLPredictionResponse:
    """
    Predict topics for text using a trained topic modeling model
    
    - **text**: Text to analyze for topics
    - **model_id**: ID of the trained topic modeling model
    - **top_n**: Number of top topics to return
    """
    
    try:
        # Make prediction
        prediction = await predict_topics(
            text=request.text,
            model_id=request.model_id,
            top_n=request.top_n
        )
        
        # Create response
        response = MLPredictionResponse(
            model_id=prediction.model_id,
            prediction=prediction.prediction,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            feature_importance=prediction.feature_importance,
            prediction_timestamp=prediction.prediction_timestamp.isoformat()
        )
        
        logger.info(f"Topic prediction completed for model: {request.model_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in topic prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting topics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during topic prediction")


@router.get("/models", response_model=List[MLModelResponse])
async def list_models_endpoint(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    current_user: Dict[str, str] = Depends(get_current_user)
) -> List[MLModelResponse]:
    """
    List all trained models with optional filtering
    
    - **model_type**: Filter by model type (neural_network, random_forest, gradient_boosting, clustering, topic_modeling)
    """
    
    try:
        # Get all models
        models = await list_ml_models()
        
        # Filter by type if specified
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        # Convert to response models
        models_response = [
            MLModelResponse(
                model_id=model.model_id,
                model_type=model.model_type,
                model_name=model.model_name,
                version=model.version,
                accuracy=model.accuracy,
                precision=model.precision,
                recall=model.recall,
                f1_score=model.f1_score,
                training_data_size=model.training_data_size,
                features_count=model.features_count,
                created_at=model.created_at.isoformat(),
                last_trained=model.last_trained.isoformat(),
                is_active=model.is_active
            )
            for model in models
        ]
        
        return models_response
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during model listing")


@router.get("/models/{model_id}", response_model=MLModelResponse)
async def get_model_info_endpoint(
    model_id: str,
    current_user: Dict[str, str] = Depends(get_current_user)
) -> MLModelResponse:
    """
    Get information about a specific trained model
    
    - **model_id**: ID of the model to retrieve
    """
    
    try:
        # Get model info
        model_info = await get_ml_model_info(model_id)
        
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Create response
        response = MLModelResponse(
            model_id=model_info.model_id,
            model_type=model_info.model_type,
            model_name=model_info.model_name,
            version=model_info.version,
            accuracy=model_info.accuracy,
            precision=model_info.precision,
            recall=model_info.recall,
            f1_score=model_info.f1_score,
            training_data_size=model_info.training_data_size,
            features_count=model_info.features_count,
            created_at=model_info.created_at.isoformat(),
            last_trained=model_info.last_trained.isoformat(),
            is_active=model_info.is_active
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during model info retrieval")


@router.get("/metrics")
async def get_ml_metrics_endpoint(
    current_user: Dict[str, str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get ML engine metrics"""
    
    try:
        metrics = await get_ml_engine_metrics()
        
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting ML metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during metrics retrieval")


@router.get("/model-types")
async def get_model_types() -> Dict[str, Any]:
    """Get available model types and their descriptions"""
    
    model_types = {
        "neural_network": {
            "description": "Deep neural network classifier with multiple hidden layers",
            "best_for": ["Complex text classification", "High accuracy requirements", "Large datasets"],
            "advantages": ["High accuracy", "Handles complex patterns", "Feature learning"],
            "disadvantages": ["Requires more data", "Longer training time", "Black box"],
            "training_time": "Medium to High",
            "accuracy": "High"
        },
        "random_forest": {
            "description": "Ensemble method using multiple decision trees",
            "best_for": ["General classification", "Feature importance", "Robust predictions"],
            "advantages": ["Fast training", "Feature importance", "Robust to overfitting"],
            "disadvantages": ["Lower accuracy than neural networks", "Memory intensive"],
            "training_time": "Low to Medium",
            "accuracy": "Medium to High"
        },
        "gradient_boosting": {
            "description": "Ensemble method that builds models sequentially",
            "best_for": ["High accuracy", "Small to medium datasets", "Feature importance"],
            "advantages": ["High accuracy", "Feature importance", "Handles missing data"],
            "disadvantages": ["Can overfit", "Longer training time", "Memory intensive"],
            "training_time": "Medium",
            "accuracy": "High"
        },
        "clustering": {
            "description": "Unsupervised learning to group similar content",
            "best_for": ["Content organization", "Pattern discovery", "Data exploration"],
            "advantages": ["No labels required", "Discovers hidden patterns", "Content grouping"],
            "disadvantages": ["No prediction labels", "Requires manual interpretation"],
            "training_time": "Low",
            "accuracy": "N/A (uses silhouette score)"
        },
        "topic_modeling": {
            "description": "Discovers hidden topics in text collections",
            "best_for": ["Content analysis", "Topic discovery", "Document organization"],
            "advantages": ["Discovers topics", "No labels required", "Interpretable results"],
            "disadvantages": ["Requires manual interpretation", "Topic quality varies"],
            "training_time": "Low to Medium",
            "accuracy": "N/A (uses coherence score)"
        }
    }
    
    return {
        "model_types": model_types,
        "total_types": len(model_types),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/capabilities")
async def get_ml_capabilities() -> Dict[str, Any]:
    """Get ML engine capabilities and features"""
    
    capabilities = {
        "training_capabilities": {
            "content_classification": "Train models to classify content into predefined categories",
            "content_clustering": "Group similar content without predefined categories",
            "topic_modeling": "Discover hidden topics in text collections",
            "custom_models": "Train custom models for specific use cases"
        },
        "prediction_capabilities": {
            "class_prediction": "Predict content class with confidence scores",
            "cluster_prediction": "Assign content to clusters with similarity scores",
            "topic_prediction": "Identify topics in content with probability scores",
            "batch_prediction": "Process multiple texts efficiently"
        },
        "model_types": {
            "neural_networks": "Deep learning models for complex pattern recognition",
            "ensemble_methods": "Random forest and gradient boosting for robust predictions",
            "unsupervised_learning": "Clustering and topic modeling without labels",
            "transfer_learning": "Pre-trained embeddings for better performance"
        },
        "features": {
            "automatic_embeddings": "Generate text embeddings using sentence transformers",
            "model_persistence": "Save and load trained models",
            "model_metrics": "Comprehensive evaluation metrics",
            "batch_processing": "Efficient processing of large datasets",
            "real_time_prediction": "Fast prediction for real-time applications"
        },
        "evaluation_metrics": {
            "classification": ["accuracy", "precision", "recall", "f1_score"],
            "clustering": ["silhouette_score", "inertia", "calinski_harabasz_score"],
            "topic_modeling": ["coherence_score", "perplexity", "topic_diversity"]
        }
    }
    
    return {
        "capabilities": capabilities,
        "total_capabilities": sum(len(cap) for cap in capabilities.values()),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health")
async def ml_health_check() -> Dict[str, Any]:
    """Health check endpoint for ML service"""
    
    try:
        health_status = await get_ml_engine_health()
        
        return {
            "status": "healthy" if health_status["status"] == "healthy" else "unhealthy",
            "service": "ml-engine",
            "timestamp": datetime.now().isoformat(),
            "ml_engine": health_status
        }
        
    except Exception as e:
        logger.error(f"ML health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "ml-engine",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# Startup and shutdown handlers
@router.on_event("startup")
async def startup_ml_service():
    """Initialize ML service on startup"""
    try:
        await initialize_content_ml_engine()
        logger.info("ML service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ML service: {e}")


@router.on_event("shutdown")
async def shutdown_ml_service():
    """Shutdown ML service on shutdown"""
    try:
        logger.info("ML service shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown ML service: {e}")




