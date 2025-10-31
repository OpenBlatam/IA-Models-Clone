"""
Machine Learning API Routes for Email Sequence System

This module provides ML-powered endpoints for predictive modeling,
recommendation systems, and automated optimization.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse

from .schemas import ErrorResponse
from ..core.machine_learning_engine import (
    ml_engine,
    ModelType,
    PredictionType,
    ModelMetrics,
    PredictionResult
)
from ..core.dependencies import get_engine, get_current_user
from ..core.exceptions import MachineLearningError

logger = logging.getLogger(__name__)

# ML router
ml_router = APIRouter(
    prefix="/api/v1/ml",
    tags=["Machine Learning"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


# Model Training Endpoints
@ml_router.post(
    "/models/churn-prediction/{sequence_id}/train",
    response_model=ModelMetrics,
    summary="Train Churn Prediction Model",
    description="Train a machine learning model to predict subscriber churn"
)
async def train_churn_prediction_model(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    training_data: List[Dict[str, Any]] = Body(..., description="Training data with features and labels"),
    user_id: str = Depends(get_current_user)
) -> ModelMetrics:
    """Train a churn prediction model for a sequence"""
    try:
        if not training_data:
            raise HTTPException(status_code=400, detail="Training data is required")
        
        metrics = await ml_engine.train_churn_prediction_model(
            sequence_id=sequence_id,
            training_data=training_data
        )
        
        logger.info(f"Churn prediction model trained for sequence {sequence_id}")
        return metrics
        
    except MachineLearningError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error training churn prediction model: {e}")
        raise HTTPException(status_code=500, detail="Failed to train churn prediction model")


@ml_router.post(
    "/models/engagement-prediction/{sequence_id}/train",
    response_model=ModelMetrics,
    summary="Train Engagement Prediction Model",
    description="Train a machine learning model to predict subscriber engagement"
)
async def train_engagement_prediction_model(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    training_data: List[Dict[str, Any]] = Body(..., description="Training data with engagement features"),
    user_id: str = Depends(get_current_user)
) -> ModelMetrics:
    """Train an engagement prediction model for a sequence"""
    try:
        if not training_data:
            raise HTTPException(status_code=400, detail="Training data is required")
        
        metrics = await ml_engine.train_engagement_prediction_model(
            sequence_id=sequence_id,
            training_data=training_data
        )
        
        logger.info(f"Engagement prediction model trained for sequence {sequence_id}")
        return metrics
        
    except MachineLearningError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error training engagement prediction model: {e}")
        raise HTTPException(status_code=500, detail="Failed to train engagement prediction model")


@ml_router.post(
    "/models/recommendation/{sequence_id}/train",
    response_model=ModelMetrics,
    summary="Train Recommendation Model",
    description="Train a machine learning model for content recommendations"
)
async def train_recommendation_model(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    interaction_data: List[Dict[str, Any]] = Body(..., description="User interaction data"),
    user_id: str = Depends(get_current_user)
) -> ModelMetrics:
    """Train a recommendation model for a sequence"""
    try:
        if not interaction_data:
            raise HTTPException(status_code=400, detail="Interaction data is required")
        
        metrics = await ml_engine.train_recommendation_model(
            sequence_id=sequence_id,
            interaction_data=interaction_data
        )
        
        logger.info(f"Recommendation model trained for sequence {sequence_id}")
        return metrics
        
    except MachineLearningError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error training recommendation model: {e}")
        raise HTTPException(status_code=500, detail="Failed to train recommendation model")


# Prediction Endpoints
@ml_router.post(
    "/predictions/churn/{sequence_id}",
    response_model=PredictionResult,
    summary="Predict Subscriber Churn",
    description="Predict churn probability for a subscriber"
)
async def predict_churn(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    subscriber_data: Dict[str, Any] = Body(..., description="Subscriber data for prediction"),
    user_id: str = Depends(get_current_user)
) -> PredictionResult:
    """Predict churn probability for a subscriber"""
    try:
        if not subscriber_data:
            raise HTTPException(status_code=400, detail="Subscriber data is required")
        
        prediction = await ml_engine.predict_churn(
            sequence_id=sequence_id,
            subscriber_data=subscriber_data
        )
        
        return prediction
        
    except MachineLearningError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting churn: {e}")
        raise HTTPException(status_code=500, detail="Failed to predict churn")


@ml_router.post(
    "/predictions/engagement/{sequence_id}",
    response_model=PredictionResult,
    summary="Predict Subscriber Engagement",
    description="Predict engagement score for a subscriber"
)
async def predict_engagement(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    subscriber_data: Dict[str, Any] = Body(..., description="Subscriber data for prediction"),
    user_id: str = Depends(get_current_user)
) -> PredictionResult:
    """Predict engagement score for a subscriber"""
    try:
        if not subscriber_data:
            raise HTTPException(status_code=400, detail="Subscriber data is required")
        
        prediction = await ml_engine.predict_engagement(
            sequence_id=sequence_id,
            subscriber_data=subscriber_data
        )
        
        return prediction
        
    except MachineLearningError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting engagement: {e}")
        raise HTTPException(status_code=500, detail="Failed to predict engagement")


# Recommendation Endpoints
@ml_router.post(
    "/recommendations/content/{sequence_id}",
    summary="Get Content Recommendations",
    description="Get ML-powered content recommendations for a subscriber"
)
async def get_content_recommendations(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    subscriber_data: Dict[str, Any] = Body(..., description="Subscriber data"),
    available_content: List[Dict[str, Any]] = Body(..., description="Available content options"),
    user_id: str = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get content recommendations for a subscriber"""
    try:
        if not subscriber_data:
            raise HTTPException(status_code=400, detail="Subscriber data is required")
        
        if not available_content:
            raise HTTPException(status_code=400, detail="Available content is required")
        
        recommendations = await ml_engine.get_content_recommendations(
            sequence_id=sequence_id,
            subscriber_data=subscriber_data,
            available_content=available_content
        )
        
        return recommendations
        
    except MachineLearningError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting content recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get content recommendations")


# Optimization Endpoints
@ml_router.post(
    "/optimization/sequence/{sequence_id}",
    summary="Optimize Sequence with ML",
    description="Use machine learning to optimize email sequences"
)
async def optimize_sequence_with_ml(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    current_sequence: Dict[str, Any] = Body(..., description="Current sequence configuration"),
    performance_data: Dict[str, Any] = Body(..., description="Historical performance data"),
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Optimize email sequence using ML insights"""
    try:
        if not current_sequence:
            raise HTTPException(status_code=400, detail="Current sequence data is required")
        
        optimization_result = await ml_engine.optimize_sequence_with_ml(
            sequence_id=sequence_id,
            current_sequence=current_sequence,
            performance_data=performance_data
        )
        
        return optimization_result
        
    except MachineLearningError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing sequence with ML: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize sequence with ML")


# Batch Prediction Endpoints
@ml_router.post(
    "/predictions/batch/churn/{sequence_id}",
    summary="Batch Churn Prediction",
    description="Predict churn for multiple subscribers"
)
async def batch_predict_churn(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    subscribers_data: List[Dict[str, Any]] = Body(..., description="List of subscriber data"),
    user_id: str = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Predict churn for multiple subscribers"""
    try:
        if not subscribers_data:
            raise HTTPException(status_code=400, detail="Subscribers data is required")
        
        predictions = []
        for subscriber_data in subscribers_data:
            try:
                prediction = await ml_engine.predict_churn(
                    sequence_id=sequence_id,
                    subscriber_data=subscriber_data
                )
                predictions.append({
                    "subscriber_id": subscriber_data.get("id"),
                    "prediction": prediction.prediction,
                    "confidence": prediction.confidence,
                    "probability": prediction.probability,
                    "feature_contributions": prediction.feature_contributions
                })
            except Exception as e:
                logger.warning(f"Failed to predict churn for subscriber {subscriber_data.get('id')}: {e}")
                predictions.append({
                    "subscriber_id": subscriber_data.get("id"),
                    "error": str(e)
                })
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in batch churn prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform batch churn prediction")


@ml_router.post(
    "/predictions/batch/engagement/{sequence_id}",
    summary="Batch Engagement Prediction",
    description="Predict engagement for multiple subscribers"
)
async def batch_predict_engagement(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    subscribers_data: List[Dict[str, Any]] = Body(..., description="List of subscriber data"),
    user_id: str = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Predict engagement for multiple subscribers"""
    try:
        if not subscribers_data:
            raise HTTPException(status_code=400, detail="Subscribers data is required")
        
        predictions = []
        for subscriber_data in subscribers_data:
            try:
                prediction = await ml_engine.predict_engagement(
                    sequence_id=sequence_id,
                    subscriber_data=subscriber_data
                )
                predictions.append({
                    "subscriber_id": subscriber_data.get("id"),
                    "prediction": prediction.prediction,
                    "confidence": prediction.confidence,
                    "feature_contributions": prediction.feature_contributions
                })
            except Exception as e:
                logger.warning(f"Failed to predict engagement for subscriber {subscriber_data.get('id')}: {e}")
                predictions.append({
                    "subscriber_id": subscriber_data.get("id"),
                    "error": str(e)
                })
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in batch engagement prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform batch engagement prediction")


# Model Management Endpoints
@ml_router.get(
    "/models/{sequence_id}/status",
    summary="Get Model Status",
    description="Get status of trained models for a sequence"
)
async def get_model_status(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get status of trained models for a sequence"""
    try:
        model_status = {
            "sequence_id": str(sequence_id),
            "models": {
                "churn_prediction": {
                    "trained": f"churn_model_{sequence_id}" in ml_engine.models,
                    "cached": await ml_engine._load_model(f"churn_model_{sequence_id}") is not None
                },
                "engagement_prediction": {
                    "trained": f"engagement_model_{sequence_id}" in ml_engine.models,
                    "cached": await ml_engine._load_model(f"engagement_model_{sequence_id}") is not None
                },
                "recommendation": {
                    "trained": f"recommendation_model_{sequence_id}" in ml_engine.models,
                    "cached": await ml_engine._load_model(f"recommendation_model_{sequence_id}") is not None
                }
            },
            "feature_importance": {
                model_key: importance 
                for model_key, importance in ml_engine.feature_importance_cache.items()
                if str(sequence_id) in model_key
            },
            "status_timestamp": datetime.utcnow().isoformat()
        }
        
        return model_status
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")


@ml_router.delete(
    "/models/{sequence_id}",
    summary="Delete Models",
    description="Delete trained models for a sequence"
)
async def delete_models(
    sequence_id: UUID = Path(..., description="Sequence ID"),
    model_types: List[str] = Query(default=["all"], description="Types of models to delete"),
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete trained models for a sequence"""
    try:
        deleted_models = []
        
        if "all" in model_types or "churn" in model_types:
            model_key = f"churn_model_{sequence_id}"
            if model_key in ml_engine.models:
                del ml_engine.models[model_key]
                deleted_models.append("churn_prediction")
        
        if "all" in model_types or "engagement" in model_types:
            model_key = f"engagement_model_{sequence_id}"
            if model_key in ml_engine.models:
                del ml_engine.models[model_key]
                deleted_models.append("engagement_prediction")
        
        if "all" in model_types or "recommendation" in model_types:
            model_key = f"recommendation_model_{sequence_id}"
            if model_key in ml_engine.models:
                del ml_engine.models[model_key]
                deleted_models.append("recommendation")
        
        return {
            "sequence_id": str(sequence_id),
            "deleted_models": deleted_models,
            "deletion_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error deleting models: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete models")


# Error handlers for ML routes
@ml_router.exception_handler(MachineLearningError)
async def ml_error_handler(request, exc):
    """Handle machine learning errors"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=f"ML service error: {exc.message}",
            error_code="ML_SERVICE_ERROR"
        ).dict()
    )






























