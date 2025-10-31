"""
Deprecated module. Use `agents.backend.onyx.server.features.ads.api.advanced` instead.

This file now re-exports the unified advanced API router to maintain backward compatibility.
"""

import warnings
from .api.advanced import router  # type: ignore

warnings.warn(
    "advanced_api is deprecated. Import from ads.api.advanced instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["router"]

@router.post("/optimize-content")
async def optimize_content(
    request: ContentOptimizationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Optimize content based on type."""
    try:
        service = AdvancedAdsService(request.app.state.httpx_client)
        result = await service.optimize_content(
            content=request.content,
            optimization_type=request.optimization_type
        )
        return format_response(result.dict())
    except Exception as e:
        raise handle_error(e)

@router.get("/audience/{segment_id}")
async def analyze_audience(
    segment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze audience segment."""
    try:
        service = AdvancedAdsService(request.app.state.httpx_client)
        result = await service.analyze_audience(segment_id)
        return format_response(result.dict())
    except Exception as e:
        raise handle_error(e)

@router.post("/brand-voice")
async def analyze_brand_voice(
    request: BrandVoiceAnalysisRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze brand voice from content samples."""
    try:
        service = AdvancedAdsService(request.app.state.httpx_client)
        result = await service.analyze_brand_voice(request.content_samples)
        return format_response(result.dict())
    except Exception as e:
        raise handle_error(e)

@router.get("/performance/{content_id}")
async def track_content_performance(
    content_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Track content performance metrics."""
    try:
        service = AdvancedAdsService(request.app.state.httpx_client)
        result = await service.track_content_performance(content_id)
        return format_response(result.dict())
    except Exception as e:
        raise handle_error(e)

@router.post("/recommendations")
async def generate_ai_recommendations(
    content: str,
    context: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate AI-powered recommendations for content."""
    try:
        service = AdvancedAdsService(request.app.state.httpx_client)
        result = await service.generate_ai_recommendations(content, context)
        return format_response({"recommendations": result})
    except Exception as e:
        raise handle_error(e)

@router.get("/impact/{content_id}")
async def analyze_content_impact(
    content_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze content impact across channels."""
    try:
        service = AdvancedAdsService(request.app.state.httpx_client)
        result = await service.analyze_content_impact(content_id)
        return format_response(result)
    except Exception as e:
        raise handle_error(e)

@router.post("/audience/optimize/{segment_id}")
async def optimize_audience_targeting(
    segment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Optimize audience targeting for a segment."""
    try:
        service = AdvancedAdsService(request.app.state.httpx_client)
        result = await service.optimize_audience_targeting(segment_id)
        return format_response(result)
    except Exception as e:
        raise handle_error(e)

@router.post("/variations")
async def generate_content_variations(
    request: ContentVariationsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate variations of content for A/B testing."""
    try:
        service = AdvancedAdsService(request.app.state.httpx_client)
        result = await service.generate_content_variations(
            content=request.content,
            variations=request.variations
        )
        return format_response({"variations": result})
    except Exception as e:
        raise handle_error(e)

@router.post("/competitor")
async def analyze_competitor_content(
    request: CompetitorAnalysisRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze competitor content and strategies."""
    try:
        service = AdvancedAdsService(request.app.state.httpx_client)
        result = await service.analyze_competitor_content(request.competitor_urls)
        return format_response(result)
    except Exception as e:
        raise handle_error(e) 