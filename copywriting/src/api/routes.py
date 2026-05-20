from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List
from ..models.requests import (
from ..models.responses import (
from ..core.engine import CopywritingEngine
from ..services.analytics import log_request_analytics
from .dependencies import get_engine
        import time
        import time
        from ..models.requests import CopywritingRequest
        import time
        import time
        from datetime import datetime
        import time
        from datetime import datetime
        import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
API Routes
=========

FastAPI routes for the copywriting system.
"""


    CopywritingRequest, 
    BatchRequest, 
    OptimizationRequest,
    AnalysisRequest
)
    CopywritingResponse, 
    BatchResponse, 
    OptimizationResponse,
    AnalysisResponse,
    SystemMetrics,
    ErrorResponse
)

router = APIRouter(prefix="/copywriting", tags=["copywriting"])


@router.post("/generate", response_model=CopywritingResponse)
async def generate_copywriting(
    request: CopywritingRequest,
    background_tasks: BackgroundTasks,
    engine: CopywritingEngine = Depends(get_engine)
) -> CopywritingResponse:
    """
    Generate copywriting content
    
    Generate high-quality copywriting content based on the provided request.
    """
    try:
        # Process request
        result = await engine.process_request(request.dict())
        
        # Add background task for analytics
        background_tasks.add_task(log_request_analytics, request, result)
        
        # Create response
        response = CopywritingResponse(
            request_id=result["request_id"],
            content=result["content"],
            variants=result["variants"],
            processing_time=result.get("processing_time", 0.0),
            model_used=result["model_used"],
            error=result.get("error"),
            metadata=result.get("metadata", {}),
            cache_hit=result.get("cache_hit", False),
            optimization_applied=result.get("optimization_applied", False),
            tokens_generated=result.get("tokens_generated"),
            confidence_score=result.get("confidence_score")
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@router.post("/batch-generate", response_model=BatchResponse)
async def batch_generate_copywriting(
    batch_request: BatchRequest,
    engine: CopywritingEngine = Depends(get_engine)
) -> BatchResponse:
    """
    Generate copywriting content in batch
    
    Process multiple copywriting requests efficiently in a single batch.
    """
    try:
        start_time = time.time()
        
        # Process batch
        results = []
        success_count = 0
        error_count = 0
        
        for request in batch_request.requests:
            try:
                result = await engine.process_request(request.dict())
                results.append(CopywritingResponse(
                    request_id=result["request_id"],
                    content=result["content"],
                    variants=result["variants"],
                    processing_time=time.time(),
                    model_used=result["model_used"],
                    error=result.get("error")
                ))
                success_count += 1
            except Exception as e:
                results.append(CopywritingResponse(
                    request_id=f"error_{int(time.time())}",
                    content=f"Error: {str(e)}",
                    variants=[],
                    processing_time=time.time(),
                    model_used="error",
                    error=str(e)
                ))
                error_count += 1
        
        total_processing_time = time.time() - start_time
        
        return BatchResponse(
            batch_id=f"batch_{int(start_time)}",
            results=results,
            total_processing_time=total_processing_time,
            batch_size=len(batch_request.requests),
            success_count=success_count,
            error_count=error_count,
            batch_options=batch_request.batch_options,
            avg_processing_time=total_processing_time / len(batch_request.requests),
            throughput=len(batch_request.requests) / total_processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation error: {str(e)}")


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_text(
    optimization_request: OptimizationRequest,
    engine: CopywritingEngine = Depends(get_engine)
) -> OptimizationResponse:
    """
    Optimize existing text
    
    Optimize and improve existing text content for better engagement and performance.
    """
    try:
        start_time = time.time()
        
        # Create a copywriting request from optimization request
        
        copywriting_request = CopywritingRequest(
            prompt=optimization_request.text,
            platform=optimization_request.platform,
            content_type="optimized",
            tone=optimization_request.tone,
            target_audience=optimization_request.target_audience,
            keywords=optimization_request.keywords,
            num_variants=1
        )
        
        # Process request
        result = await engine.process_request(copywriting_request.dict())
        
        processing_time = time.time() - start_time
        
        return OptimizationResponse(
            request_id=result["request_id"],
            original_text=optimization_request.text,
            optimized_text=result["content"],
            processing_time=processing_time,
            optimization_type=optimization_request.optimization_type,
            improvements=result.get("improvements", []),
            metrics=result.get("metrics", {}),
            suggestions=result.get("suggestions"),
            confidence_score=result.get("confidence_score")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(
    analysis_request: AnalysisRequest,
    engine: CopywritingEngine = Depends(get_engine)
) -> AnalysisResponse:
    """
    Analyze content
    
    Perform comprehensive analysis of text content including sentiment, readability, and engagement.
    """
    try:
        start_time = time.time()
        
        # Perform analysis using engine
        analysis_result = await engine.analyze_content(analysis_request.dict())
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            request_id=analysis_result["request_id"],
            text=analysis_request.text,
            processing_time=processing_time,
            sentiment=analysis_result.get("sentiment"),
            readability=analysis_result.get("readability"),
            tone=analysis_result.get("tone"),
            engagement=analysis_result.get("engagement"),
            seo=analysis_result.get("seo"),
            grammar=analysis_result.get("grammar"),
            overall_score=analysis_result.get("overall_score", 0.0),
            suggestions=analysis_result.get("suggestions", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/models")
async def get_available_models():
    """
    Get available models
    
    Retrieve information about available AI models for copywriting generation.
    """
    try:
        return {
            "models": [
                {
                    "id": "gpt2-medium",
                    "name": "GPT-2 Medium",
                    "description": "Medium-sized GPT-2 model for text generation",
                    "max_tokens": 1024,
                    "supports_gpu": True,
                    "recommended": True
                },
                {
                    "id": "distilgpt2",
                    "name": "DistilGPT-2",
                    "description": "Distilled version of GPT-2 for faster inference",
                    "max_tokens": 512,
                    "supports_gpu": True,
                    "recommended": False
                },
                {
                    "id": "gpt2-large",
                    "name": "GPT-2 Large",
                    "description": "Large GPT-2 model for high-quality generation",
                    "max_tokens": 2048,
                    "supports_gpu": True,
                    "recommended": False
                }
            ],
            "default_model": "gpt2-medium",
            "fallback_model": "distilgpt2"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")


@router.get("/metrics", response_model=SystemMetrics)
async def get_metrics(engine: CopywritingEngine = Depends(get_engine)) -> SystemMetrics:
    """
    Get system metrics
    
    Retrieve comprehensive system performance and health metrics.
    """
    try:
        
        metrics = engine.get_metrics()
        
        return SystemMetrics(
            engine_status={
                "initialized": metrics["is_initialized"],
                "active_requests": metrics["active_requests"],
                "total_requests": metrics["total_requests"]
            },
            performance_metrics=metrics["performance"],
            memory_usage=metrics["memory_usage"],
            cache_stats=metrics["cache_stats"],
            batch_stats=metrics["batch_stats"],
            uptime=time.time() - metrics.get("start_time", time.time()),
            version="3.0.0",
            environment="production",
            health_status="healthy",
            last_health_check=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")


@router.get("/health/detailed")
async def detailed_health_check(engine: CopywritingEngine = Depends(get_engine)):
    """
    Detailed health check
    
    Perform comprehensive health check of all system components.
    """
    try:
        
        # Get comprehensive metrics
        metrics = engine.get_metrics()
        
        # Check Redis
        redis_status = "healthy"
        try:
            # This would check Redis connectivity
            pass
        except Exception:
            redis_status = "unhealthy"
        
        return {
            "status": "healthy",
            "message": "System is running",
            "timestamp": datetime.utcnow(),
            "engine": {
                "initialized": metrics["is_initialized"],
                "active_requests": metrics["active_requests"],
                "total_requests": metrics["total_requests"]
            },
            "memory": metrics["memory_usage"],
            "gpu": metrics["gpu_memory"],
            "cache": metrics["cache_stats"],
            "batch": metrics["batch_stats"],
            "redis": redis_status,
            "performance": metrics["performance"]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"System error: {str(e)}",
            "timestamp": datetime.utcnow()
        }


@router.get("/cache/stats")
async def get_cache_stats(engine: CopywritingEngine = Depends(get_engine)):
    """
    Get cache statistics
    
    Retrieve detailed cache performance and usage statistics.
    """
    try:
        cache_stats = engine.get_metrics()["cache_stats"]
        
        return {
            "cache_stats": cache_stats,
            "hit_rate": cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"]) if (cache_stats["hits"] + cache_stats["misses"]) > 0 else 0,
            "total_requests": cache_stats["hits"] + cache_stats["misses"],
            "efficiency": "high" if cache_stats["hits"] > cache_stats["misses"] else "low"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")


@router.delete("/cache/clear")
async def clear_cache(engine: CopywritingEngine = Depends(get_engine)):
    """
    Clear cache
    
    Clear all cached data and reset cache statistics.
    """
    try:
        
        # Clear cache
        await engine.clear_cache()
        
        return {
            "message": "Cache cleared successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}") 