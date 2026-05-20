"""
Ultra Fast NLP API
==================

API REST ultra-rápida para máximo rendimiento
y velocidad de procesamiento.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging

from .ultra_fast_nlp import ultra_fast_nlp, UltraFastResult

logger = logging.getLogger(__name__)

# Create ultra-fast router
router = APIRouter(prefix="/ultra-fast", tags=["Ultra Fast NLP"])

# Ultra-fast request/response models
class UltraFastRequest(BaseModel):
    """Request model for ultra-fast analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=10000)
    language: Optional[str] = Field(default="en", description="Language code")
    use_cache: bool = Field(default=True, description="Use ultra-fast caching")
    parallel_processing: bool = Field(default=True, description="Use parallel processing")

class UltraFastResponse(BaseModel):
    """Response model for ultra-fast analysis."""
    text: str
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    processing_time: float
    cache_hit: bool
    timestamp: datetime

class UltraFastBatchRequest(BaseModel):
    """Request model for ultra-fast batch analysis."""
    texts: List[str] = Field(..., description="Texts to analyze", min_items=1, max_items=1000)
    language: Optional[str] = Field(default="en", description="Language code")
    use_cache: bool = Field(default=True, description="Use ultra-fast caching")
    parallel_processing: bool = Field(default=True, description="Use parallel processing")
    batch_size: Optional[int] = Field(default=128, description="Batch size for processing", ge=1, le=512)

class UltraFastBatchResponse(BaseModel):
    """Response model for ultra-fast batch analysis."""
    results: List[UltraFastResponse]
    total_processing_time: float
    average_processing_time: float
    success_count: int
    error_count: int
    cache_hit_rate: float
    throughput_per_second: float
    batch_size: int
    timestamp: datetime

class UltraFastStatusResponse(BaseModel):
    """Response model for ultra-fast system status."""
    system: Dict[str, Any]
    performance: Dict[str, Any]
    cache: Dict[str, Any]
    memory: Dict[str, Any]
    timestamp: datetime

# Dependency to ensure ultra-fast NLP system is initialized
async def get_ultra_fast_nlp_system():
    """Get initialized ultra-fast NLP system."""
    if not ultra_fast_nlp.is_initialized:
        await ultra_fast_nlp.initialize()
    return ultra_fast_nlp

# Ultra-fast API Endpoints

@router.get("/health", response_model=UltraFastStatusResponse)
async def get_ultra_fast_health():
    """Get ultra-fast system health status."""
    try:
        status = await ultra_fast_nlp.get_ultra_fast_status()
        return UltraFastStatusResponse(**status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@router.post("/analyze", response_model=UltraFastResponse)
async def analyze_ultra_fast(request: UltraFastRequest):
    """Perform ultra-fast text analysis."""
    try:
        result = await ultra_fast_nlp.analyze_ultra_fast(
            text=request.text,
            language=request.language or "en",
            use_cache=request.use_cache,
            parallel_processing=request.parallel_processing
        )
        
        return UltraFastResponse(
            text=result.text,
            sentiment=result.sentiment,
            entities=result.entities,
            keywords=result.keywords,
            processing_time=result.processing_time,
            cache_hit=result.cache_hit,
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Ultra-fast analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-fast analysis failed: {e}")

@router.post("/batch", response_model=UltraFastBatchResponse)
async def batch_analyze_ultra_fast(request: UltraFastBatchRequest):
    """Perform ultra-fast batch analysis."""
    try:
        start_time = datetime.now()
        
        # Apply batch size if specified
        if request.batch_size:
            ultra_fast_nlp.config.batch_size = request.batch_size
        
        results = await ultra_fast_nlp.batch_analyze_ultra_fast(
            texts=request.texts,
            language=request.language or "en",
            use_cache=request.use_cache,
            parallel_processing=request.parallel_processing
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate statistics
        success_count = len([r for r in results if r.processing_time > 0])
        error_count = len(results) - success_count
        cache_hits = len([r for r in results if r.cache_hit])
        cache_hit_rate = cache_hits / len(results) if results else 0
        throughput = len(results) / total_time if total_time > 0 else 0
        
        # Convert to response format
        response_results = []
        for result in results:
            response_results.append(UltraFastResponse(
                text=result.text,
                sentiment=result.sentiment,
                entities=result.entities,
                keywords=result.keywords,
                processing_time=result.processing_time,
                cache_hit=result.cache_hit,
                timestamp=result.timestamp
            ))
        
        return UltraFastBatchResponse(
            results=response_results,
            total_processing_time=total_time,
            average_processing_time=total_time / len(results) if results else 0,
            success_count=success_count,
            error_count=error_count,
            cache_hit_rate=cache_hit_rate,
            throughput_per_second=throughput,
            batch_size=request.batch_size or 128,
            timestamp=start_time
        )
        
    except Exception as e:
        logger.error(f"Ultra-fast batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-fast batch analysis failed: {e}")

@router.get("/status", response_model=UltraFastStatusResponse)
async def get_ultra_fast_status():
    """Get ultra-fast system status."""
    try:
        status = await ultra_fast_nlp.get_ultra_fast_status()
        return UltraFastStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get ultra-fast status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra-fast status: {e}")

@router.get("/metrics", response_model=Dict[str, Any])
async def get_ultra_fast_metrics():
    """Get ultra-fast system metrics."""
    try:
        status = await ultra_fast_nlp.get_ultra_fast_status()
        
        # Performance metrics
        performance = status.get('performance', {})
        
        # System metrics
        system = status.get('system', {})
        
        # Cache metrics
        cache = status.get('cache', {})
        
        # Memory metrics
        memory = status.get('memory', {})
        
        return {
            'performance': performance,
            'system': system,
            'cache': cache,
            'memory': memory,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get ultra-fast metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra-fast metrics: {e}")

@router.post("/stress-test")
async def run_ultra_fast_stress_test(
    concurrent_requests: int = Query(50, description="Number of concurrent requests", ge=1, le=200),
    text_length: int = Query(100, description="Text length for testing", ge=10, le=1000)
):
    """Run ultra-fast stress test."""
    try:
        # Generate test text
        test_text = "This is an ultra-fast stress test for maximum performance. " * (text_length // 50)
        
        start_time = datetime.now()
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent_requests):
            task = ultra_fast_nlp.analyze_ultra_fast(
                text=f"{test_text} Request #{i+1}",
                use_cache=True,
                parallel_processing=True
            )
            tasks.append(task)
        
        # Execute stress test
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Analyze results
        success_count = len([r for r in results if not isinstance(r, Exception)])
        error_count = len(results) - success_count
        throughput = concurrent_requests / total_time if total_time > 0 else 0
        
        stress_test_results = {
            'concurrent_requests': concurrent_requests,
            'text_length': text_length,
            'total_time': total_time,
            'throughput_per_second': throughput,
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': (success_count / concurrent_requests) * 100,
            'average_time_per_request': total_time / concurrent_requests,
            'ultra_fast_mode': ultra_fast_nlp.config.ultra_fast_mode,
            'timestamp': start_time.isoformat()
        }
        
        return stress_test_results
        
    except Exception as e:
        logger.error(f"Ultra-fast stress test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-fast stress test failed: {e}")

@router.get("/benchmark")
async def run_ultra_fast_benchmark():
    """Run ultra-fast benchmark."""
    try:
        # Test texts of different lengths
        test_texts = [
            "Short text.",
            "This is a medium length text for testing ultra-fast performance.",
            "This is a longer text that will test the ultra-fast system's ability to handle more complex analysis while maintaining maximum speed and efficiency."
        ]
        
        start_time = datetime.now()
        
        # Test each text
        results = []
        for text in test_texts:
            result = await ultra_fast_nlp.analyze_ultra_fast(
                text=text,
                use_cache=True,
                parallel_processing=True
            )
            results.append(result)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate benchmark metrics
        processing_times = [r.processing_time for r in results]
        cache_hits = len([r for r in results if r.cache_hit])
        
        benchmark_results = {
            'total_time': total_time,
            'average_processing_time': sum(processing_times) / len(processing_times),
            'min_processing_time': min(processing_times),
            'max_processing_time': max(processing_times),
            'cache_hit_rate': cache_hits / len(results),
            'throughput_per_second': len(results) / total_time,
            'ultra_fast_mode': ultra_fast_nlp.config.ultra_fast_mode,
            'timestamp': start_time.isoformat()
        }
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Ultra-fast benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-fast benchmark failed: {e}")

@router.get("/capabilities", response_model=Dict[str, Any])
async def get_ultra_fast_capabilities():
    """Get ultra-fast system capabilities."""
    return {
        "ultra_fast_mode": True,
        "parallel_processing": True,
        "gpu_acceleration": ultra_fast_nlp.gpu_available,
        "intelligent_caching": True,
        "memory_optimization": True,
        "background_optimization": True,
        "analysis_tasks": [
            "sentiment_analysis",
            "entity_extraction",
            "keyword_extraction"
        ],
        "supported_languages": ["en"],
        "max_text_length": 10000,
        "max_batch_size": 1000,
        "max_concurrent_requests": 200,
        "cache_enabled": True,
        "ultra_fast_optimization": True
    }

# Utility endpoints
@router.get("/supported-languages", response_model=List[str])
async def get_supported_languages():
    """Get list of supported languages."""
    return ["en"]

@router.get("/analysis-tasks", response_model=List[str])
async def get_analysis_tasks():
    """Get available analysis tasks."""
    return ["sentiment", "entities", "keywords"]












