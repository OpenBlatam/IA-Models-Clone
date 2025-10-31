"""
Machine Learning API routes for Facebook Posts API
ML-powered predictions, content scoring, and trend analysis
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
import structlog

from ..core.config import get_settings
from ..core.models import PostRequest, PostStatus, ContentType, AudienceType, OptimizationLevel
from ..api.schemas import (
    PostUpdateRequest, BatchPostRequest, OptimizationRequest,
    ErrorResponse, SystemHealth, PerformanceMetrics
)
from ..api.dependencies import (
    get_facebook_engine, get_current_user, check_rate_limit,
    get_request_id, validate_post_id
)
from ..services.ml_service import get_ml_service, PredictionResult, ContentScore, TrendPrediction
from ..services.optimization_service import get_optimization_service, OptimizationStrategy, TestType
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.monitoring import get_monitor, timed
from ..infrastructure.database import get_db_manager, PostRepository

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/ml", tags=["Machine Learning"])


# ML Prediction Routes

@router.post(
    "/predict/engagement",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Engagement prediction generated successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "ML prediction error"}
    },
    summary="Predict content engagement",
    description="Predict engagement rate for Facebook post content using ML models"
)
@timed("ml_engagement_prediction")
async def predict_engagement(
    content: str = Query(..., description="Content to analyze", min_length=10),
    content_type: Optional[str] = Query(None, description="Content type"),
    audience_type: Optional[str] = Query(None, description="Target audience type"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Predict engagement rate for content"""
    
    if len(content) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content too long for analysis (max 5000 characters)"
        )
    
    try:
        # Prepare metadata
        metadata = {}
        if content_type:
            metadata["content_type"] = content_type
        if audience_type:
            metadata["audience_type"] = audience_type
        
        # Get ML service
        ml_service = get_ml_service()
        
        # Predict engagement
        prediction = await ml_service.predict_engagement(content, metadata)
        
        logger.info(
            "Engagement prediction completed",
            content_length=len(content),
            predicted_value=prediction.predicted_value,
            confidence=prediction.confidence,
            model_used=prediction.model_used,
            request_id=request_id
        )
        
        return {
            "success": True,
            "prediction": {
                "engagement_rate": prediction.predicted_value,
                "confidence": prediction.confidence,
                "model_used": prediction.model_used,
                "processing_time": prediction.processing_time
            },
            "metadata": {
                "content_length": len(content),
                "features_used": prediction.features_used,
                "model_metadata": prediction.metadata
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Engagement prediction failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Engagement prediction failed: {str(e)}"
        )


@router.post(
    "/predict/reach",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Reach prediction generated successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "ML prediction error"}
    },
    summary="Predict content reach",
    description="Predict reach potential for Facebook post content using ML models"
)
@timed("ml_reach_prediction")
async def predict_reach(
    content: str = Query(..., description="Content to analyze", min_length=10),
    content_type: Optional[str] = Query(None, description="Content type"),
    audience_type: Optional[str] = Query(None, description="Target audience type"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Predict reach potential for content"""
    
    if len(content) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content too long for analysis (max 5000 characters)"
        )
    
    try:
        # Prepare metadata
        metadata = {}
        if content_type:
            metadata["content_type"] = content_type
        if audience_type:
            metadata["audience_type"] = audience_type
        
        # Get ML service
        ml_service = get_ml_service()
        
        # Predict reach
        prediction = await ml_service.predict_reach(content, metadata)
        
        logger.info(
            "Reach prediction completed",
            content_length=len(content),
            predicted_value=prediction.predicted_value,
            confidence=prediction.confidence,
            model_used=prediction.model_used,
            request_id=request_id
        )
        
        return {
            "success": True,
            "prediction": {
                "reach_potential": prediction.predicted_value,
                "confidence": prediction.confidence,
                "model_used": prediction.model_used,
                "processing_time": prediction.processing_time
            },
            "metadata": {
                "content_length": len(content),
                "features_used": prediction.features_used,
                "model_metadata": prediction.metadata
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Reach prediction failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reach prediction failed: {str(e)}"
        )


@router.post(
    "/predict/clicks",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Click prediction generated successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "ML prediction error"}
    },
    summary="Predict content clicks",
    description="Predict click rate for Facebook post content using ML models"
)
@timed("ml_click_prediction")
async def predict_clicks(
    content: str = Query(..., description="Content to analyze", min_length=10),
    content_type: Optional[str] = Query(None, description="Content type"),
    audience_type: Optional[str] = Query(None, description="Target audience type"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Predict click rate for content"""
    
    if len(content) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content too long for analysis (max 5000 characters)"
        )
    
    try:
        # Prepare metadata
        metadata = {}
        if content_type:
            metadata["content_type"] = content_type
        if audience_type:
            metadata["audience_type"] = audience_type
        
        # Get ML service
        ml_service = get_ml_service()
        
        # Predict clicks
        prediction = await ml_service.predict_clicks(content, metadata)
        
        logger.info(
            "Click prediction completed",
            content_length=len(content),
            predicted_value=prediction.predicted_value,
            confidence=prediction.confidence,
            model_used=prediction.model_used,
            request_id=request_id
        )
        
        return {
            "success": True,
            "prediction": {
                "click_rate": prediction.predicted_value,
                "confidence": prediction.confidence,
                "model_used": prediction.model_used,
                "processing_time": prediction.processing_time
            },
            "metadata": {
                "content_length": len(content),
                "features_used": prediction.features_used,
                "model_metadata": prediction.metadata
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Click prediction failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Click prediction failed: {str(e)}"
        )


@router.post(
    "/score/content",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Content score generated successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "ML scoring error"}
    },
    summary="Score content quality",
    description="Generate comprehensive content quality scores using ML models"
)
@timed("ml_content_scoring")
async def score_content(
    content: str = Query(..., description="Content to score", min_length=10),
    content_type: Optional[str] = Query(None, description="Content type"),
    audience_type: Optional[str] = Query(None, description="Target audience type"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Score content quality using ML models"""
    
    if len(content) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content too long for analysis (max 5000 characters)"
        )
    
    try:
        # Prepare metadata
        metadata = {}
        if content_type:
            metadata["content_type"] = content_type
        if audience_type:
            metadata["audience_type"] = audience_type
        
        # Get ML service
        ml_service = get_ml_service()
        
        # Score content
        score = await ml_service.score_content(content, metadata)
        
        logger.info(
            "Content scoring completed",
            content_length=len(content),
            overall_score=score.overall_score,
            engagement_score=score.engagement_score,
            request_id=request_id
        )
        
        return {
            "success": True,
            "scores": {
                "overall_score": score.overall_score,
                "engagement_score": score.engagement_score,
                "reach_score": score.reach_score,
                "quality_score": score.quality_score,
                "virality_score": score.virality_score
            },
            "factors": score.factors,
            "recommendations": score.recommendations,
            "metadata": {
                "content_length": len(content),
                "content_type": content_type,
                "audience_type": audience_type
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Content scoring failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content scoring failed: {str(e)}"
        )


@router.post(
    "/predict/trend",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Trend prediction generated successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "ML prediction error"}
    },
    summary="Predict content trends",
    description="Predict trend potential and virality for topics using ML models"
)
@timed("ml_trend_prediction")
async def predict_trend(
    topic: str = Query(..., description="Topic to analyze", min_length=3),
    content_type: str = Query(..., description="Content type"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Predict trend potential for a topic"""
    
    if len(topic) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topic too long (max 100 characters)"
        )
    
    # Validate content type
    valid_content_types = [content_type.value for content_type in ContentType]
    if content_type not in valid_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type. Valid types: {valid_content_types}"
        )
    
    try:
        # Get ML service
        ml_service = get_ml_service()
        
        # Predict trend
        prediction = await ml_service.predict_trend(topic, content_type)
        
        logger.info(
            "Trend prediction completed",
            topic=topic,
            content_type=content_type,
            predicted_engagement=prediction.predicted_engagement,
            confidence=prediction.confidence,
            request_id=request_id
        )
        
        return {
            "success": True,
            "prediction": {
                "topic": prediction.topic,
                "predicted_engagement": prediction.predicted_engagement,
                "confidence": prediction.confidence,
                "time_to_peak_hours": prediction.time_to_peak,
                "peak_engagement": prediction.peak_engagement,
                "duration_hours": prediction.duration
            },
            "factors": prediction.factors,
            "metadata": {
                "content_type": content_type,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Trend prediction failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trend prediction failed: {str(e)}"
        )


# Optimization Routes

@router.post(
    "/optimize/content",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Content optimization completed successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Optimization error"}
    },
    summary="Optimize content",
    description="Optimize content using advanced optimization strategies"
)
@timed("ml_content_optimization")
async def optimize_content(
    content: str = Query(..., description="Content to optimize", min_length=10),
    strategy: str = Query(..., description="Optimization strategy"),
    audience_type: str = Query(..., description="Target audience type"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Optimize content using advanced strategies"""
    
    if len(content) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content too long for optimization (max 5000 characters)"
        )
    
    # Validate strategy
    valid_strategies = [strategy.value for strategy in OptimizationStrategy]
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid optimization strategy. Valid strategies: {valid_strategies}"
        )
    
    # Validate audience type
    valid_audience_types = [audience_type.value for audience_type in AudienceType]
    if audience_type not in valid_audience_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audience type. Valid types: {valid_audience_types}"
        )
    
    try:
        # Get optimization service
        optimization_service = get_optimization_service()
        
        # Optimize content
        result = await optimization_service.optimize_content(
            content,
            OptimizationStrategy(strategy),
            AudienceType(audience_type)
        )
        
        logger.info(
            "Content optimization completed",
            strategy=strategy,
            audience_type=audience_type,
            expected_improvement=result.expected_improvement,
            request_id=request_id
        )
        
        return {
            "success": True,
            "optimization": {
                "original_content": result.original_content,
                "optimized_content": result.optimized_content,
                "optimization_type": result.optimization_type,
                "expected_improvement": result.expected_improvement,
                "confidence_score": result.confidence_score,
                "changes_made": result.changes_made,
                "processing_time": result.processing_time
            },
            "metadata": {
                "strategy": strategy,
                "audience_type": audience_type,
                "optimization_metadata": result.metadata
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Content optimization failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content optimization failed: {str(e)}"
        )


@router.post(
    "/ab-test/create",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "A/B test created successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "A/B test creation error"}
    },
    summary="Create A/B test",
    description="Create A/B test for content optimization"
)
@timed("ml_ab_test_creation")
async def create_ab_test(
    test_type: str = Query(..., description="Type of A/B test"),
    original_variant: Dict[str, Any] = Query(..., description="Original variant data"),
    test_variant: Dict[str, Any] = Query(..., description="Test variant data"),
    target_metric: str = Query(..., description="Target metric to measure"),
    test_duration_days: int = Query(7, description="Test duration in days", ge=1, le=30),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create A/B test for content optimization"""
    
    # Validate test type
    valid_test_types = [test_type.value for test_type in TestType]
    if test_type not in valid_test_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid test type. Valid types: {valid_test_types}"
        )
    
    # Validate target metric
    valid_metrics = ["engagement", "reach", "clicks", "shares", "comments", "sentiment"]
    if target_metric not in valid_metrics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid target metric. Valid metrics: {valid_metrics}"
        )
    
    try:
        # Get optimization service
        optimization_service = get_optimization_service()
        
        # Create A/B test
        test = await optimization_service.create_ab_test(
            TestType(test_type),
            original_variant,
            test_variant,
            target_metric,
            test_duration_days
        )
        
        logger.info(
            "A/B test created",
            test_id=test.test_id,
            test_type=test_type,
            target_metric=target_metric,
            duration_days=test_duration_days,
            request_id=request_id
        )
        
        return {
            "success": True,
            "test": {
                "test_id": test.test_id,
                "test_type": test.test_type.value,
                "target_metric": test.target_metric,
                "test_duration_days": test.test_duration_days,
                "status": test.status,
                "created_at": test.created_at.isoformat()
            },
            "metadata": {
                "original_variant": test.original_variant,
                "test_variant": test.test_variant
            },
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "A/B test creation failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A/B test creation failed: {str(e)}"
        )


@router.get(
    "/ab-test/{test_id}/variant",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Test variant retrieved successfully"},
        404: {"description": "A/B test not found"},
        500: {"description": "Variant retrieval error"}
    },
    summary="Get A/B test variant",
    description="Get test variant for user (A/B split)"
)
@timed("ml_ab_test_variant")
async def get_ab_test_variant(
    test_id: str = Path(..., description="A/B test ID"),
    user_id: str = Query(..., description="User ID for variant assignment"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get A/B test variant for user"""
    
    try:
        # Get optimization service
        optimization_service = get_optimization_service()
        
        # Get test variant
        variant = await optimization_service.get_test_variant(test_id, user_id)
        
        logger.info(
            "A/B test variant retrieved",
            test_id=test_id,
            user_id=user_id,
            request_id=request_id
        )
        
        return {
            "success": True,
            "test_id": test_id,
            "user_id": user_id,
            "variant": variant,
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "A/B test variant retrieval failed",
            test_id=test_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A/B test variant retrieval failed: {str(e)}"
        )


@router.post(
    "/ab-test/{test_id}/metric",
    responses={
        200: {"description": "Test metric recorded successfully"},
        404: {"description": "A/B test not found"},
        500: {"description": "Metric recording error"}
    },
    summary="Record A/B test metric",
    description="Record metric value for A/B test"
)
@timed("ml_ab_test_metric")
async def record_ab_test_metric(
    test_id: str = Path(..., description="A/B test ID"),
    variant: str = Query(..., description="Test variant (original or test)"),
    metric_value: float = Query(..., description="Metric value to record"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Record metric for A/B test"""
    
    # Validate variant
    if variant not in ["original", "test"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Variant must be 'original' or 'test'"
        )
    
    try:
        # Get optimization service
        optimization_service = get_optimization_service()
        
        # Record metric
        await optimization_service.record_test_metric(test_id, variant, metric_value)
        
        logger.info(
            "A/B test metric recorded",
            test_id=test_id,
            variant=variant,
            metric_value=metric_value,
            request_id=request_id
        )
        
        return {
            "success": True,
            "test_id": test_id,
            "variant": variant,
            "metric_value": metric_value,
            "recorded_at": datetime.now().isoformat(),
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(
            "A/B test metric recording failed",
            test_id=test_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A/B test metric recording failed: {str(e)}"
        )


@router.get(
    "/ab-test/{test_id}/results",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "A/B test results retrieved successfully"},
        404: {"description": "A/B test not found"},
        500: {"description": "Results analysis error"}
    },
    summary="Get A/B test results",
    description="Get analyzed results for A/B test"
)
@timed("ml_ab_test_results")
async def get_ab_test_results(
    test_id: str = Path(..., description="A/B test ID"),
    request_id: str = Depends(get_request_id),
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get A/B test results"""
    
    try:
        # Get optimization service
        optimization_service = get_optimization_service()
        
        # Analyze test results
        results = await optimization_service.analyze_test_results(test_id)
        
        logger.info(
            "A/B test results retrieved",
            test_id=test_id,
            request_id=request_id
        )
        
        return {
            "success": True,
            "test_id": test_id,
            "results": results,
            "request_id": request_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "A/B test results retrieval failed",
            test_id=test_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A/B test results retrieval failed: {str(e)}"
        )


# Export router
__all__ = ["router"]






























