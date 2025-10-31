"""
Advanced API Routes
==================

Advanced endpoints for AI engine, analytics, and content optimization.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse

from ..schemas import (
    CopywritingRequest,
    CopywritingResponse,
    CopywritingVariant,
    ErrorResponse
)
from ..services import get_copywriting_service, CopywritingService
from ..exceptions import ContentGenerationError, ValidationError
from .ai_engine import AIProvider, AIProviderConfig, ai_engine_manager
from .analytics import (
    analytics_engine,
    TimeRange,
    PerformanceMetrics,
    QualityMetrics,
    UsageMetrics,
    EngagementMetrics
)
from .content_optimizer import (
    content_optimizer,
    OptimizationStrategy,
    OptimizationGoal,
    OptimizationResult,
    ABTestResult,
    ContentInsight
)

logger = logging.getLogger(__name__)

# Create advanced router
advanced_router = APIRouter(
    prefix="/api/v2/copywriting/advanced",
    tags=["advanced"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)


# AI Engine Routes
@advanced_router.post(
    "/ai/configure",
    summary="Configure AI Provider",
    description="Configure an AI provider for content generation"
)
async def configure_ai_provider(
    provider: AIProvider,
    api_key: str,
    model: str,
    max_tokens: int = 2000,
    temperature: float = 0.7
):
    """Configure an AI provider"""
    try:
        config = AIProviderConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        ai_engine_manager.configure_provider(config)
        
        return {
            "message": f"AI provider {provider} configured successfully",
            "provider": provider,
            "model": model
        }
        
    except Exception as e:
        logger.error(f"Failed to configure AI provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure AI provider: {str(e)}"
        )


@advanced_router.get(
    "/ai/providers",
    summary="Get Available AI Providers",
    description="Get list of configured AI providers"
)
async def get_ai_providers():
    """Get available AI providers"""
    try:
        providers = ai_engine_manager.get_available_providers()
        health_status = await ai_engine_manager.health_check_all()
        
        return {
            "available_providers": [provider.value for provider in providers],
            "health_status": {provider.value: status for provider, status in health_status.items()}
        }
        
    except Exception as e:
        logger.error(f"Failed to get AI providers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI providers: {str(e)}"
        )


@advanced_router.post(
    "/ai/generate",
    response_model=CopywritingResponse,
    summary="Generate Content with AI",
    description="Generate content using configured AI providers"
)
async def generate_with_ai(
    request: CopywritingRequest,
    preferred_provider: Optional[AIProvider] = None,
    service: CopywritingService = Depends(get_copywriting_service)
):
    """Generate content using AI providers"""
    try:
        logger.info(f"Generating content with AI provider: {preferred_provider}")
        
        variants = await ai_engine_manager.generate_content(request, preferred_provider)
        
        response = CopywritingResponse(
            variants=variants,
            processing_time_ms=1000,  # Mock processing time
            metadata={
                "ai_generated": True,
                "provider": preferred_provider.value if preferred_provider else "auto",
                "tone": request.tone,
                "style": request.style,
                "purpose": request.purpose
            }
        )
        
        logger.info(f"Generated {len(variants)} AI variants")
        return response
        
    except ContentGenerationError as e:
        logger.error(f"AI content generation failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Unexpected error in AI generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during AI generation"
        )


# Analytics Routes
@advanced_router.get(
    "/analytics/performance",
    summary="Get Performance Analytics",
    description="Get performance metrics and analytics"
)
async def get_performance_analytics(
    time_range: TimeRange = Query(default=TimeRange.DAY, description="Time range for analytics"),
    service: CopywritingService = Depends(get_copywriting_service)
):
    """Get performance analytics"""
    try:
        async with await service.get_session() as session:
            metrics = await analytics_engine.get_performance_metrics(session, time_range)
            
            return {
                "time_range": time_range,
                "metrics": {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "average_response_time_ms": metrics.average_response_time_ms,
                    "p95_response_time_ms": metrics.p95_response_time_ms,
                    "p99_response_time_ms": metrics.p99_response_time_ms,
                    "requests_per_minute": metrics.requests_per_minute,
                    "error_rate": metrics.error_rate,
                    "cache_hit_rate": metrics.cache_hit_rate
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get performance analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance analytics"
        )


@advanced_router.get(
    "/analytics/quality",
    summary="Get Quality Analytics",
    description="Get content quality metrics and analytics"
)
async def get_quality_analytics(
    time_range: TimeRange = Query(default=TimeRange.DAY, description="Time range for analytics"),
    service: CopywritingService = Depends(get_copywriting_service)
):
    """Get quality analytics"""
    try:
        async with await service.get_session() as session:
            metrics = await analytics_engine.get_quality_metrics(session, time_range)
            
            return {
                "time_range": time_range,
                "metrics": {
                    "average_confidence_score": metrics.average_confidence_score,
                    "average_rating": metrics.average_rating,
                    "total_feedback_count": metrics.total_feedback_count,
                    "positive_feedback_rate": metrics.positive_feedback_rate,
                    "improvement_suggestions_count": metrics.improvement_suggestions_count,
                    "most_common_improvements": metrics.most_common_improvements
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get quality analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve quality analytics"
        )


@advanced_router.get(
    "/analytics/usage",
    summary="Get Usage Analytics",
    description="Get usage patterns and analytics"
)
async def get_usage_analytics(
    time_range: TimeRange = Query(default=TimeRange.DAY, description="Time range for analytics"),
    service: CopywritingService = Depends(get_copywriting_service)
):
    """Get usage analytics"""
    try:
        async with await service.get_session() as session:
            metrics = await analytics_engine.get_usage_metrics(session, time_range)
            
            return {
                "time_range": time_range,
                "metrics": {
                    "total_variants_generated": metrics.total_variants_generated,
                    "average_variants_per_request": metrics.average_variants_per_request,
                    "most_popular_tones": metrics.most_popular_tones,
                    "most_popular_styles": metrics.most_popular_styles,
                    "most_popular_purposes": metrics.most_popular_purposes,
                    "average_word_count": metrics.average_word_count,
                    "cta_inclusion_rate": metrics.cta_inclusion_rate
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get usage analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage analytics"
        )


@advanced_router.get(
    "/analytics/dashboard",
    summary="Get Comprehensive Dashboard",
    description="Get comprehensive analytics dashboard data"
)
async def get_analytics_dashboard(
    time_range: TimeRange = Query(default=TimeRange.DAY, description="Time range for analytics"),
    service: CopywritingService = Depends(get_copywriting_service)
):
    """Get comprehensive analytics dashboard"""
    try:
        async with await service.get_session() as session:
            dashboard_data = await analytics_engine.get_comprehensive_dashboard(session, time_range)
            
            return dashboard_data
            
    except Exception as e:
        logger.error(f"Failed to get analytics dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics dashboard"
        )


# Content Optimization Routes
@advanced_router.post(
    "/optimize",
    summary="Optimize Content",
    description="Optimize content using various strategies"
)
async def optimize_content(
    variant: CopywritingVariant,
    request: CopywritingRequest,
    strategy: OptimizationStrategy,
    goal: OptimizationGoal = OptimizationGoal.CONVERSION_RATE
):
    """Optimize content using specified strategy"""
    try:
        logger.info(f"Optimizing content with strategy: {strategy}")
        
        result = await content_optimizer.optimize_content(
            variant, request, strategy, goal
        )
        
        return {
            "optimization_result": {
                "strategy": result.optimization_strategy,
                "improvement_score": result.improvement_score,
                "confidence_boost": result.confidence_boost,
                "changes_made": result.changes_made,
                "original_variant": {
                    "title": result.original_variant.title,
                    "content": result.original_variant.content,
                    "confidence_score": result.original_variant.confidence_score
                },
                "optimized_variant": {
                    "title": result.optimized_variant.title,
                    "content": result.optimized_variant.content,
                    "confidence_score": result.optimized_variant.confidence_score
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Content optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize content"
        )


@advanced_router.post(
    "/optimize/batch",
    summary="Batch Optimize Content",
    description="Optimize multiple content variants with multiple strategies"
)
async def batch_optimize_content(
    variants: List[CopywritingVariant],
    request: CopywritingRequest,
    strategies: List[OptimizationStrategy]
):
    """Batch optimize content variants"""
    try:
        logger.info(f"Batch optimizing {len(variants)} variants with {len(strategies)} strategies")
        
        results = await content_optimizer.batch_optimize(variants, request, strategies)
        
        return {
            "batch_optimization_results": [
                {
                    "strategy": result.optimization_strategy,
                    "improvement_score": result.improvement_score,
                    "confidence_boost": result.confidence_boost,
                    "changes_made": result.changes_made
                }
                for result in results
            ],
            "total_results": len(results),
            "best_result": {
                "strategy": results[0].optimization_strategy if results else None,
                "improvement_score": results[0].improvement_score if results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Batch optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to batch optimize content"
        )


@advanced_router.post(
    "/ab-test",
    summary="Run A/B Test",
    description="Run A/B test between two content variants"
)
async def run_ab_test(
    variant_a: CopywritingVariant,
    variant_b: CopywritingVariant,
    test_duration_hours: int = 24,
    target_metric: OptimizationGoal = OptimizationGoal.CONVERSION_RATE
):
    """Run A/B test between two variants"""
    try:
        logger.info(f"Running A/B test for {test_duration_hours} hours")
        
        result = await content_optimizer.run_ab_test(
            variant_a, variant_b, test_duration_hours, target_metric
        )
        
        return {
            "ab_test_result": {
                "test_id": result.test_id,
                "winner": result.winner,
                "confidence_level": result.confidence_level,
                "sample_size": result.sample_size,
                "statistical_significance": result.statistical_significance,
                "metrics": result.metrics,
                "variant_a": {
                    "title": result.variant_a.title,
                    "confidence_score": result.variant_a.confidence_score
                },
                "variant_b": {
                    "title": result.variant_b.title,
                    "confidence_score": result.variant_b.confidence_score
                }
            }
        }
        
    except Exception as e:
        logger.error(f"A/B test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run A/B test"
        )


@advanced_router.get(
    "/insights/{variant_id}",
    summary="Get Content Insights",
    description="Get insights and recommendations for content"
)
async def get_content_insights(
    variant_id: UUID,
    request: CopywritingRequest,
    historical_data: Optional[List[Dict[str, Any]]] = None
):
    """Get content insights and recommendations"""
    try:
        # In a real implementation, you would fetch the variant from the database
        # For now, we'll create a mock variant
        variant = CopywritingVariant(
            id=variant_id,
            title="Sample Content",
            content="This is sample content for analysis.",
            word_count=10,
            confidence_score=0.8
        )
        
        insights = await content_optimizer.get_content_insights(
            variant, request, historical_data
        )
        
        return {
            "variant_id": variant_id,
            "insights": [
                {
                    "insight_type": insight.insight_type,
                    "description": insight.description,
                    "impact_score": insight.impact_score,
                    "recommendation": insight.recommendation,
                    "implementation_difficulty": insight.implementation_difficulty,
                    "expected_improvement": insight.expected_improvement
                }
                for insight in insights
            ],
            "total_insights": len(insights),
            "high_impact_insights": len([i for i in insights if i.impact_score > 0.7])
        }
        
    except Exception as e:
        logger.error(f"Failed to get content insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve content insights"
        )


@advanced_router.get(
    "/optimization/strategies",
    summary="Get Optimization Strategies",
    description="Get available optimization strategies"
)
async def get_optimization_strategies():
    """Get available optimization strategies"""
    return {
        "strategies": [
            {
                "name": strategy.value,
                "description": _get_strategy_description(strategy)
            }
            for strategy in OptimizationStrategy
        ],
        "goals": [
            {
                "name": goal.value,
                "description": _get_goal_description(goal)
            }
            for goal in OptimizationGoal
        ]
    }


def _get_strategy_description(strategy: OptimizationStrategy) -> str:
    """Get description for optimization strategy"""
    descriptions = {
        OptimizationStrategy.A_B_TESTING: "Compare two variants to find the best performer",
        OptimizationStrategy.KEYWORD_OPTIMIZATION: "Optimize keyword usage and density",
        OptimizationStrategy.READABILITY_IMPROVEMENT: "Improve content readability and clarity",
        OptimizationStrategy.ENGAGEMENT_BOOST: "Increase user engagement and interaction",
        OptimizationStrategy.CONVERSION_OPTIMIZATION: "Optimize for higher conversion rates",
        OptimizationStrategy.TONE_ADJUSTMENT: "Adjust content tone to match requirements"
    }
    return descriptions.get(strategy, "Optimization strategy")


def _get_goal_description(goal: OptimizationGoal) -> str:
    """Get description for optimization goal"""
    descriptions = {
        OptimizationGoal.CLICK_THROUGH_RATE: "Maximize click-through rates",
        OptimizationGoal.CONVERSION_RATE: "Maximize conversion rates",
        OptimizationGoal.ENGAGEMENT_TIME: "Increase user engagement time",
        OptimizationGoal.SHARE_RATE: "Increase content sharing",
        OptimizationGoal.FEEDBACK_SCORE: "Improve user feedback scores",
        OptimizationGoal.READABILITY: "Improve content readability"
    }
    return descriptions.get(goal, "Optimization goal")






























