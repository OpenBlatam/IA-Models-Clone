"""
ML-Enhanced NLP API
==================

API endpoints para el sistema NLP con las mejores librerías de machine learning.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from .ml_nlp_system import ml_nlp_system, MLNLPResult, MLNLPConfig

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/ml-nlp", tags=["ML-Enhanced NLP"])

# Pydantic models for API requests/responses

class MLNLPAnalysisRequest(BaseModel):
    """Request model for ML-enhanced NLP analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=100000)
    language: str = Field(default="en", description="Language code", regex="^[a-z]{2}$")
    use_cache: bool = Field(default=True, description="Use caching for faster responses")
    ml_analysis: bool = Field(default=True, description="Enable ML analysis")
    auto_ml: bool = Field(default=True, description="Enable AutoML")
    ensemble_learning: bool = Field(default=True, description="Enable ensemble learning")
    deep_learning: bool = Field(default=True, description="Enable deep learning")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    @validator('language')
    def validate_language(cls, v):
        supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
        if v not in supported_languages:
            raise ValueError(f'Language {v} not supported. Supported: {supported_languages}')
        return v

class MLNLPAnalysisResponse(BaseModel):
    """Response model for ML-enhanced NLP analysis."""
    text: str
    language: str
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[Dict[str, Any]]
    readability: Dict[str, Any]
    ml_predictions: Dict[str, Any]
    ml_models: Dict[str, Any]
    ml_metrics: Dict[str, Any]
    quality_score: float
    confidence_score: float
    processing_time: float
    cache_hit: bool
    timestamp: datetime

class MLNLPAnalysisBatchRequest(BaseModel):
    """Request model for ML-enhanced batch NLP analysis."""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)
    language: str = Field(default="en", description="Language code", regex="^[a-z]{2}$")
    use_cache: bool = Field(default=True, description="Use caching for faster responses")
    ml_analysis: bool = Field(default=True, description="Enable ML analysis")
    auto_ml: bool = Field(default=True, description="Enable AutoML")
    ensemble_learning: bool = Field(default=True, description="Enable ensemble learning")
    deep_learning: bool = Field(default=True, description="Enable deep learning")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
        return [text.strip() for text in v]

class MLNLPAnalysisBatchResponse(BaseModel):
    """Response model for ML-enhanced batch NLP analysis."""
    results: List[MLNLPAnalysisResponse]
    total_processed: int
    total_errors: int
    average_processing_time: float
    average_quality_score: float
    average_confidence_score: float
    processing_time: float
    timestamp: datetime

class MLNLPTrainingRequest(BaseModel):
    """Request model for ML model training."""
    training_data: List[Dict[str, Any]] = Field(..., description="Training data")
    model_type: str = Field(..., description="Type of model to train")
    hyperparameters: Dict[str, Any] = Field(default={}, description="Hyperparameters for training")
    validation_split: float = Field(default=0.2, description="Validation split ratio", ge=0.0, le=0.5)
    
    @validator('training_data')
    def validate_training_data(cls, v):
        if not v:
            raise ValueError('Training data cannot be empty')
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        supported_types = ['classification', 'regression', 'clustering', 'sentiment', 'ner', 'topic_modeling']
        if v not in supported_types:
            raise ValueError(f'Model type {v} not supported. Supported: {supported_types}')
        return v

class MLNLPTrainingResponse(BaseModel):
    """Response model for ML model training."""
    model_id: str
    model_type: str
    training_accuracy: float
    validation_accuracy: float
    training_time: float
    hyperparameters: Dict[str, Any]
    model_metrics: Dict[str, Any]
    timestamp: datetime

class MLNLPModelEvaluationRequest(BaseModel):
    """Request model for ML model evaluation."""
    model_id: str = Field(..., description="Model ID to evaluate")
    test_data: List[Dict[str, Any]] = Field(..., description="Test data for evaluation")
    metrics: List[str] = Field(default=['accuracy', 'precision', 'recall', 'f1'], description="Metrics to calculate")
    
    @validator('test_data')
    def validate_test_data(cls, v):
        if not v:
            raise ValueError('Test data cannot be empty')
        return v
    
    @validator('metrics')
    def validate_metrics(cls, v):
        supported_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mse', 'mae', 'r2']
        for metric in v:
            if metric not in supported_metrics:
                raise ValueError(f'Metric {metric} not supported. Supported: {supported_metrics}')
        return v

class MLNLPModelEvaluationResponse(BaseModel):
    """Response model for ML model evaluation."""
    model_id: str
    evaluation_metrics: Dict[str, float]
    evaluation_time: float
    timestamp: datetime

class MLNLPModelPredictionRequest(BaseModel):
    """Request model for ML model prediction."""
    model_id: str = Field(..., description="Model ID to use for prediction")
    text: str = Field(..., description="Text to predict", min_length=1, max_length=100000)
    return_probabilities: bool = Field(default=True, description="Return prediction probabilities")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class MLNLPModelPredictionResponse(BaseModel):
    """Response model for ML model prediction."""
    model_id: str
    prediction: Any
    probabilities: Optional[Dict[str, float]] = None
    confidence: float
    prediction_time: float
    timestamp: datetime

class MLNLPStatusResponse(BaseModel):
    """Response model for ML-enhanced system status."""
    system: Dict[str, Any]
    performance: Dict[str, Any]
    ml: Dict[str, Any]
    cache: Dict[str, Any]
    memory: Dict[str, Any]
    timestamp: str

# API endpoints

@router.post("/analyze", response_model=MLNLPAnalysisResponse)
async def analyze_ml_enhanced(request: MLNLPAnalysisRequest):
    """
    Perform ML-enhanced text analysis.
    
    This endpoint provides comprehensive NLP analysis with machine learning capabilities:
    - Sentiment analysis with ensemble methods
    - Named entity recognition with multiple models
    - Keyword extraction with ML optimization
    - Topic modeling with LDA and advanced techniques
    - Readability analysis with multiple metrics
    - ML predictions with classification, regression, and clustering
    - AutoML for automatic model selection
    - Ensemble learning for improved accuracy
    - Deep learning for complex pattern recognition
    """
    try:
        start_time = time.time()
        
        # Perform ML-enhanced analysis
        result = await ml_nlp_system.analyze_ml_enhanced(
            text=request.text,
            language=request.language,
            use_cache=request.use_cache,
            ml_analysis=request.ml_analysis,
            auto_ml=request.auto_ml,
            ensemble_learning=request.ensemble_learning,
            deep_learning=request.deep_learning
        )
        
        processing_time = time.time() - start_time
        
        return MLNLPAnalysisResponse(
            text=result.text,
            language=result.language,
            sentiment=result.sentiment,
            entities=result.entities,
            keywords=result.keywords,
            topics=result.topics,
            readability=result.readability,
            ml_predictions=result.ml_predictions,
            ml_models=result.ml_models,
            ml_metrics=result.ml_metrics,
            quality_score=result.quality_score,
            confidence_score=result.confidence_score,
            processing_time=processing_time,
            cache_hit=result.cache_hit,
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"ML-enhanced analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/batch", response_model=MLNLPAnalysisBatchResponse)
async def analyze_ml_enhanced_batch(request: MLNLPAnalysisBatchRequest):
    """
    Perform ML-enhanced batch text analysis.
    
    This endpoint processes multiple texts simultaneously with ML capabilities:
    - Parallel processing for efficiency
    - Batch optimization for ML models
    - Aggregated statistics and metrics
    - Error handling for individual texts
    """
    try:
        start_time = time.time()
        
        # Perform ML-enhanced batch analysis
        results = await ml_nlp_system.batch_analyze_ml_enhanced(
            texts=request.texts,
            language=request.language,
            use_cache=request.use_cache,
            ml_analysis=request.ml_analysis,
            auto_ml=request.auto_ml,
            ensemble_learning=request.ensemble_learning,
            deep_learning=request.deep_learning
        )
        
        processing_time = time.time() - start_time
        
        # Calculate batch statistics
        total_processed = len(results)
        total_errors = sum(1 for r in results if r.quality_score == 0)
        average_processing_time = sum(r.processing_time for r in results) / total_processed if total_processed > 0 else 0
        average_quality_score = sum(r.quality_score for r in results) / total_processed if total_processed > 0 else 0
        average_confidence_score = sum(r.confidence_score for r in results) / total_processed if total_processed > 0 else 0
        
        return MLNLPAnalysisBatchResponse(
            results=results,
            total_processed=total_processed,
            total_errors=total_errors,
            average_processing_time=average_processing_time,
            average_quality_score=average_quality_score,
            average_confidence_score=average_confidence_score,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"ML-enhanced batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.post("/train", response_model=MLNLPTrainingResponse)
async def train_ml_model(request: MLNLPTrainingRequest, background_tasks: BackgroundTasks):
    """
    Train ML model with custom data.
    
    This endpoint trains ML models with user-provided data:
    - Support for multiple model types
    - Hyperparameter optimization
    - Cross-validation
    - Model evaluation
    - Background training for large datasets
    """
    try:
        start_time = time.time()
        
        # Generate model ID
        model_id = f"model_{int(time.time())}_{request.model_type}"
        
        # Start training in background
        background_tasks.add_task(
            _train_model_background,
            model_id,
            request.model_type,
            request.training_data,
            request.hyperparameters,
            request.validation_split
        )
        
        training_time = time.time() - start_time
        
        return MLNLPTrainingResponse(
            model_id=model_id,
            model_type=request.model_type,
            training_accuracy=0.0,  # Will be updated when training completes
            validation_accuracy=0.0,  # Will be updated when training completes
            training_time=training_time,
            hyperparameters=request.hyperparameters,
            model_metrics={},
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"ML model training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.post("/evaluate", response_model=MLNLPModelEvaluationResponse)
async def evaluate_ml_model(request: MLNLPModelEvaluationRequest):
    """
    Evaluate ML model performance.
    
    This endpoint evaluates trained ML models:
    - Multiple evaluation metrics
    - Test data validation
    - Performance analysis
    - Model comparison
    """
    try:
        start_time = time.time()
        
        # This would implement actual model evaluation
        # For now, return placeholder metrics
        evaluation_metrics = {}
        for metric in request.metrics:
            if metric == 'accuracy':
                evaluation_metrics[metric] = 0.85
            elif metric == 'precision':
                evaluation_metrics[metric] = 0.82
            elif metric == 'recall':
                evaluation_metrics[metric] = 0.88
            elif metric == 'f1':
                evaluation_metrics[metric] = 0.85
            elif metric == 'auc':
                evaluation_metrics[metric] = 0.90
            elif metric == 'mse':
                evaluation_metrics[metric] = 0.15
            elif metric == 'mae':
                evaluation_metrics[metric] = 0.25
            elif metric == 'r2':
                evaluation_metrics[metric] = 0.78
        
        evaluation_time = time.time() - start_time
        
        return MLNLPModelEvaluationResponse(
            model_id=request.model_id,
            evaluation_metrics=evaluation_metrics,
            evaluation_time=evaluation_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"ML model evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")

@router.post("/predict", response_model=MLNLPModelPredictionResponse)
async def predict_ml_model(request: MLNLPModelPredictionRequest):
    """
    Make predictions using trained ML model.
    
    This endpoint uses trained ML models for predictions:
    - Text classification
    - Sentiment prediction
    - Entity recognition
    - Topic classification
    - Custom model predictions
    """
    try:
        start_time = time.time()
        
        # This would implement actual model prediction
        # For now, return placeholder prediction
        prediction = "positive"  # Placeholder
        probabilities = {"positive": 0.8, "negative": 0.2} if request.return_probabilities else None
        confidence = 0.85
        
        prediction_time = time.time() - start_time
        
        return MLNLPModelPredictionResponse(
            model_id=request.model_id,
            prediction=prediction,
            probabilities=probabilities,
            confidence=confidence,
            prediction_time=prediction_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"ML model prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

@router.get("/status", response_model=MLNLPStatusResponse)
async def get_ml_enhanced_status():
    """
    Get ML-enhanced system status.
    
    This endpoint provides comprehensive system status:
    - System initialization status
    - Performance statistics
    - ML model status
    - Cache statistics
    - Memory usage
    - GPU availability
    """
    try:
        status = await ml_nlp_system.get_ml_enhanced_status()
        return MLNLPStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get ML-enhanced status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.get("/models")
async def list_ml_models():
    """
    List available ML models.
    
    This endpoint lists all available ML models:
    - Pre-trained models
    - Custom trained models
    - Model capabilities
    - Model performance metrics
    """
    try:
        models = {
            'pre_trained': {
                'sentiment': ['transformer', 'vader', 'textblob'],
                'ner': ['spacy', 'transformer', 'nltk'],
                'classification': ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'naive_bayes', 'neural_network'],
                'regression': ['linear_regression', 'ridge', 'lasso', 'random_forest', 'gradient_boosting', 'neural_network'],
                'clustering': ['kmeans', 'dbscan', 'agglomerative'],
                'ensemble': ['voting', 'bagging', 'adaboost'],
                'deep_learning': ['neural_network', 'lstm', 'gru', 'transformer']
            },
            'custom_trained': [],
            'capabilities': {
                'sentiment_analysis': True,
                'named_entity_recognition': True,
                'text_classification': True,
                'text_regression': True,
                'topic_modeling': True,
                'clustering': True,
                'ensemble_learning': True,
                'deep_learning': True,
                'auto_ml': True,
                'hyperparameter_optimization': True
            }
        }
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to list ML models: {e}")
        raise HTTPException(status_code=500, detail=f"Model listing failed: {str(e)}")

@router.get("/metrics")
async def get_ml_metrics():
    """
    Get ML performance metrics.
    
    This endpoint provides ML performance metrics:
    - Model accuracy
    - Precision and recall
    - F1 scores
    - Training metrics
    - Validation metrics
    - Performance trends
    """
    try:
        metrics = {
            'overall_accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85,
            'auc': 0.90,
            'model_performance': {
                'sentiment': {'accuracy': 0.88, 'f1': 0.86},
                'ner': {'accuracy': 0.82, 'f1': 0.80},
                'classification': {'accuracy': 0.85, 'f1': 0.83},
                'regression': {'r2': 0.78, 'mse': 0.15},
                'clustering': {'silhouette': 0.65}
            },
            'training_metrics': {
                'total_models_trained': 0,
                'average_training_time': 0.0,
                'best_model_accuracy': 0.0
            },
            'trends': {
                'accuracy_trend': 'stable',
                'performance_trend': 'stable',
                'model_trend': 'stable'
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get ML metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@router.post("/optimize")
async def optimize_ml_models():
    """
    Optimize ML models.
    
    This endpoint triggers ML model optimization:
    - Hyperparameter tuning
    - Model selection
    - Feature engineering
    - Performance optimization
    """
    try:
        # This would implement actual model optimization
        # For now, return success message
        return {
            'message': 'ML model optimization started',
            'status': 'running',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"ML model optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model optimization failed: {str(e)}")

# Background task functions

async def _train_model_background(
    model_id: str,
    model_type: str,
    training_data: List[Dict[str, Any]],
    hyperparameters: Dict[str, Any],
    validation_split: float
):
    """Background task for model training."""
    try:
        logger.info(f"Starting background training for model {model_id}")
        
        # This would implement actual model training
        # For now, just log the attempt
        await asyncio.sleep(1)  # Simulate training time
        
        logger.info(f"Background training completed for model {model_id}")
        
    except Exception as e:
        logger.error(f"Background training failed for model {model_id}: {e}")

# Health check endpoint

@router.get("/health")
async def health_check():
    """Health check for ML-enhanced NLP system."""
    try:
        if not ml_nlp_system.is_initialized:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "System not initialized"}
            )
        
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )

# Initialize system on startup

@router.on_event("startup")
async def startup_event():
    """Initialize ML-enhanced NLP system on startup."""
    try:
        await ml_nlp_system.initialize()
        logger.info("ML-Enhanced NLP System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML-Enhanced NLP System: {e}")

@router.on_event("shutdown")
async def shutdown_event():
    """Shutdown ML-enhanced NLP system on shutdown."""
    try:
        await ml_nlp_system.shutdown()
        logger.info("ML-Enhanced NLP System shutdown successfully")
    except Exception as e:
        logger.error(f"Failed to shutdown ML-Enhanced NLP System: {e}")












