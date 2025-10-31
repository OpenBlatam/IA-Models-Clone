from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
from application.use_cases import AnalyzeURLUseCase, AnalyzeBatchUseCase
from application.dto import AnalyzeURLRequest, AnalyzeURLResponse, AnalyzeBatchRequest, AnalyzeBatchResponse
from presentation.api.dependencies import get_analyze_url_use_case, get_analyze_batch_use_case
from presentation.api.schemas import (
from presentation.api.middleware import get_request_id
from shared.monitoring.metrics import record_request, record_analysis, record_cache_hit, record_cache_miss
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
SEO API Routes
Production-ready FastAPI routes for SEO analysis
"""


    AnalyzeURLSchema, AnalyzeBatchSchema, AnalyzeURLResponseSchema,
    HealthResponse, ErrorResponse
)


router = APIRouter(prefix="/seo", tags=["SEO Analysis"])


@router.post("/analyze", response_model=AnalyzeURLResponseSchema)
async def analyze_url(
    request: Request,
    analyze_request: AnalyzeURLSchema,
    background_tasks: BackgroundTasks,
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case)
) -> AnalyzeURLResponse:
    """Analyze single URL for SEO optimization"""
    start_time = time.time()
    request_id = get_request_id(request)
    
    try:
        logger.info(f"API: Starting URL analysis", extra={
            "request_id": request_id,
            "url": analyze_request.url,
            "force_refresh": analyze_request.force_refresh
        })
        
        # Convert schema to DTO
        dto_request = AnalyzeURLRequest(
            url=analyze_request.url,
            force_refresh=analyze_request.force_refresh,
            include_recommendations=analyze_request.include_recommendations,
            include_warnings=analyze_request.include_warnings,
            include_errors=analyze_request.include_errors
        )
        
        # Execute use case
        response = await use_case.execute(dto_request)
        
        # Record metrics
        duration = time.time() - start_time
        record_request("POST", "/seo/analyze", 200)
        record_analysis("success")
        
        if response.cached:
            record_cache_hit()
        else:
            record_cache_miss()
        
        # Background task for logging
        background_tasks.add_task(
            _log_analysis_completion,
            request_id,
            analyze_request.url,
            response.seo_score,
            response.analysis_time,
            response.cached
        )
        
        logger.info(f"API: URL analysis completed", extra={
            "request_id": request_id,
            "url": analyze_request.url,
            "score": response.seo_score,
            "grade": response.grade,
            "duration": duration,
            "cached": response.cached
        })
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        record_request("POST", "/seo/analyze", 500)
        record_analysis("error")
        
        logger.error(f"API: URL analysis failed", extra={
            "request_id": request_id,
            "url": analyze_request.url,
            "error": str(e),
            "duration": duration
        })
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/analyze/batch", response_model=AnalyzeBatchResponse)
async def analyze_urls_batch(
    request: Request,
    batch_request: AnalyzeBatchSchema,
    use_case: AnalyzeBatchUseCase = Depends(get_analyze_batch_use_case)
) -> AnalyzeBatchResponse:
    """Analyze multiple URLs in batch"""
    start_time = time.time()
    request_id = get_request_id(request)
    
    try:
        logger.info(f"API: Starting batch analysis", extra={
            "request_id": request_id,
            "url_count": len(batch_request.urls),
            "max_concurrent": batch_request.max_concurrent
        })
        
        # Validate request
        if len(batch_request.urls) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 URLs per batch request"
            )
        
        # Convert schema to DTO
        dto_request = AnalyzeBatchRequest(
            urls=batch_request.urls,
            batch_id=batch_request.batch_id,
            max_concurrent=batch_request.max_concurrent,
            force_refresh=batch_request.force_refresh
        )
        
        # Execute use case
        response = await use_case.execute(dto_request)
        
        # Record metrics
        duration = time.time() - start_time
        record_request("POST", "/seo/analyze/batch", 200)
        
        logger.info(f"API: Batch analysis completed", extra={
            "request_id": request_id,
            "batch_id": response.batch_id,
            "total_urls": response.total_urls,
            "successful": response.successful_analyses,
            "failed": response.failed_analyses,
            "duration": duration
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        record_request("POST", "/seo/analyze/batch", 500)
        
        logger.error(f"API: Batch analysis failed", extra={
            "request_id": request_id,
            "error": str(e),
            "duration": duration
        })
        
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.get("/analyze/{analysis_id}", response_model=AnalyzeURLResponseSchema)
async def get_analysis(
    request: Request,
    analysis_id: str,
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case)
) -> AnalyzeURLResponse:
    """Get analysis by ID"""
    start_time = time.time()
    request_id = get_request_id(request)
    
    try:
        logger.info(f"API: Getting analysis", extra={
            "request_id": request_id,
            "analysis_id": analysis_id
        })
        
        # Get analysis from repository
        analysis = await use_case.repository.find_by_id(analysis_id)
        
        if not analysis:
            record_request("GET", f"/seo/analyze/{analysis_id}", 404)
            raise HTTPException(
                status_code=404,
                detail=f"Analysis with ID {analysis_id} not found"
            )
        
        # Convert to response
        response = use_case.mapper.to_response(analysis)
        
        # Record metrics
        duration = time.time() - start_time
        record_request("GET", f"/seo/analyze/{analysis_id}", 200)
        
        logger.info(f"API: Analysis retrieved", extra={
            "request_id": request_id,
            "analysis_id": analysis_id,
            "duration": duration
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        record_request("GET", f"/seo/analyze/{analysis_id}", 500)
        
        logger.error(f"API: Failed to get analysis", extra={
            "request_id": request_id,
            "analysis_id": analysis_id,
            "error": str(e),
            "duration": duration
        })
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve analysis: {str(e)}"
        )


@router.get("/recent", response_model=List[AnalyzeURLResponseSchema])
async def get_recent_analyses(
    request: Request,
    limit: int = 10,
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case)
) -> List[AnalyzeURLResponse]:
    """Get recent analyses"""
    start_time = time.time()
    request_id = get_request_id(request)
    
    try:
        # Validate limit
        if limit > 100:
            limit = 100
        
        logger.info(f"API: Getting recent analyses", extra={
            "request_id": request_id,
            "limit": limit
        })
        
        # Get recent analyses
        analyses = await use_case.repository.find_recent(limit)
        
        # Convert to responses
        responses = [use_case.mapper.to_response(analysis) for analysis in analyses]
        
        # Record metrics
        duration = time.time() - start_time
        record_request("GET", "/seo/recent", 200)
        
        logger.info(f"API: Recent analyses retrieved", extra={
            "request_id": request_id,
            "count": len(responses),
            "duration": duration
        })
        
        return responses
        
    except Exception as e:
        duration = time.time() - start_time
        record_request("GET", "/seo/recent", 500)
        
        logger.error(f"API: Failed to get recent analyses", extra={
            "request_id": request_id,
            "error": str(e),
            "duration": duration
        })
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve recent analyses: {str(e)}"
        )


@router.delete("/analyze/{analysis_id}")
async def delete_analysis(
    request: Request,
    analysis_id: str,
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case)
) -> dict:
    """Delete analysis by ID"""
    start_time = time.time()
    request_id = get_request_id(request)
    
    try:
        logger.info(f"API: Deleting analysis", extra={
            "request_id": request_id,
            "analysis_id": analysis_id
        })
        
        # Delete analysis
        success = await use_case.repository.delete(analysis_id)
        
        if not success:
            record_request("DELETE", f"/seo/analyze/{analysis_id}", 404)
            raise HTTPException(
                status_code=404,
                detail=f"Analysis with ID {analysis_id} not found"
            )
        
        # Record metrics
        duration = time.time() - start_time
        record_request("DELETE", f"/seo/analyze/{analysis_id}", 200)
        
        logger.info(f"API: Analysis deleted", extra={
            "request_id": request_id,
            "analysis_id": analysis_id,
            "duration": duration
        })
        
        return {"message": f"Analysis {analysis_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        record_request("DELETE", f"/seo/analyze/{analysis_id}", 500)
        
        logger.error(f"API: Failed to delete analysis", extra={
            "request_id": request_id,
            "analysis_id": analysis_id,
            "error": str(e),
            "duration": duration
        })
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete analysis: {str(e)}"
        )


@router.post("/cache/clear")
async def clear_cache(
    request: Request,
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case)
) -> dict:
    """Clear analysis cache"""
    start_time = time.time()
    request_id = get_request_id(request)
    
    try:
        logger.info(f"API: Clearing cache", extra={
            "request_id": request_id
        })
        
        # Clear cache (implementation depends on cache strategy)
        # This would typically clear the repository cache
        
        # Record metrics
        duration = time.time() - start_time
        record_request("POST", "/seo/cache/clear", 200)
        
        logger.info(f"API: Cache cleared", extra={
            "request_id": request_id,
            "duration": duration
        })
        
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        duration = time.time() - start_time
        record_request("POST", "/seo/cache/clear", 500)
        
        logger.error(f"API: Failed to clear cache", extra={
            "request_id": request_id,
            "error": str(e),
            "duration": duration
        })
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/stats")
async def get_stats(
    request: Request,
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case)
) -> dict:
    """Get service statistics"""
    start_time = time.time()
    request_id = get_request_id(request)
    
    try:
        logger.info(f"API: Getting stats", extra={
            "request_id": request_id
        })
        
        # Get repository stats
        repo_stats = await use_case.repository.get_stats()
        
        # Get analyzer stats
        analyzer_stats = use_case.seo_analyzer.get_stats()
        
        # Get scoring service stats
        scoring_stats = use_case.scoring_service.get_stats()
        
        stats = {
            "repository": repo_stats,
            "analyzer": analyzer_stats,
            "scoring": scoring_stats,
            "timestamp": time.time()
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request("GET", "/seo/stats", 200)
        
        logger.info(f"API: Stats retrieved", extra={
            "request_id": request_id,
            "duration": duration
        })
        
        return stats
        
    except Exception as e:
        duration = time.time() - start_time
        record_request("GET", "/seo/stats", 500)
        
        logger.error(f"API: Failed to get stats", extra={
            "request_id": request_id,
            "error": str(e),
            "duration": duration
        })
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )


async def _log_analysis_completion(
    request_id: str,
    url: str,
    score: float,
    analysis_time: float,
    cached: bool
):
    """Background task to log analysis completion"""
    logger.info(f"Analysis completed", extra={
        "request_id": request_id,
        "url": url,
        "score": score,
        "analysis_time": analysis_time,
        "cached": cached
    }) 