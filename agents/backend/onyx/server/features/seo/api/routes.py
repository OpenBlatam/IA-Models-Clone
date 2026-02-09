from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, validator
from loguru import logger
import time
import asyncio
from ..services.seo_service_factory import get_seo_service, get_factory
from ..core.ultra_optimized_analyzer import SEOAnalysis, KeywordAnalysis
from typing import Any, List, Dict, Optional
import logging
"""
Production API routes for Ultra-Optimized SEO Service.
FastAPI routes with comprehensive validation and error handling.
"""



# Create router
router = APIRouter(prefix="/seo", tags=["SEO Analysis"])

# Request/Response models
class URLRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL to analyze")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Analysis options")

class BatchRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., description="List of URLs to analyze", max_items=100)
    options: Optional[Dict[str, Any]] = Field(default={}, description="Batch analysis options")

class ComparisonRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., description="URLs to compare", min_items=2, max_items=10)
    comparison_type: Optional[str] = Field(default="seo", description="Type of comparison")

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    uptime: float
    components: Dict[str, Any]

class MetricsResponse(BaseModel):
    requests_total: int
    requests_per_second: float
    average_response_time: float
    error_rate: float
    cache_hit_rate: float
    memory_usage: float
    cpu_usage: float

# Dependency injection
async def get_seo_service_dependency():
    """Get SEO service instance."""
    return get_seo_service()

async def get_factory_dependency():
    """Get factory instance."""
    return get_factory()

# Rate limiting and validation
def validate_url(url: str) -> str:
    """Validate and normalize URL."""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url

def validate_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Validate analysis options."""
    valid_options = {
        'include_keywords': bool,
        'include_sentiment': bool,
        'include_readability': bool,
        'use_selenium': bool,
        'timeout': int,
        'max_retries': int
    }
    
    validated = {}
    for key, value in options.items():
        if key in valid_options:
            try:
                validated[key] = valid_options[key](value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid option {key}: {value}")
    
    return validated

# Routes
@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_url(
    request: URLRequest,
    seo_service = Depends(get_seo_service_dependency)
):
    """
    Analyze a single URL for SEO.
    
    - **url**: The URL to analyze
    - **options**: Optional analysis configuration
    """
    start_time = time.perf_counter()
    
    try:
        # Validate and normalize URL
        url = str(request.url)
        options = validate_options(request.options or {})
        
        logger.info(f"Analyzing URL: {url}")
        
        # Perform analysis
        result = await seo_service.scrape(url, **options)
        
        # Calculate processing time
        processing_time = time.perf_counter() - start_time
        
        # Add timing information
        result['processing_time'] = processing_time
        result['timestamp'] = time.time()
        
        logger.info(f"Analysis completed for {url} in {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed for {request.url}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/batch", response_model=Dict[str, Any])
async def batch_analyze(
    request: BatchRequest,
    seo_service = Depends(get_seo_service_dependency)
):
    """
    Analyze multiple URLs in batch.
    
    - **urls**: List of URLs to analyze (max 100)
    - **options**: Optional batch configuration
    """
    start_time = time.perf_counter()
    
    try:
        urls = [str(url) for url in request.urls]
        options = validate_options(request.options or {})
        
        logger.info(f"Batch analyzing {len(urls)} URLs")
        
        # Perform batch analysis
        results = await seo_service.batch_analyze(urls, **options)
        
        # Calculate processing time
        processing_time = time.perf_counter() - start_time
        
        # Add batch information
        batch_result = {
            'success': True,
            'total_urls': len(urls),
            'successful_analyses': len([r for r in results if r.get('success')]),
            'failed_analyses': len([r for r in results if not r.get('success')]),
            'processing_time': processing_time,
            'average_time_per_url': processing_time / len(urls) if urls else 0,
            'results': results,
            'timestamp': time.time()
        }
        
        logger.info(f"Batch analysis completed: {batch_result['successful_analyses']}/{len(urls)} successful")
        
        return batch_result
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )

@router.post("/compare", response_model=Dict[str, Any])
async def compare_urls(
    request: ComparisonRequest,
    seo_service = Depends(get_seo_service_dependency)
):
    """
    Compare multiple URLs for SEO analysis.
    
    - **urls**: URLs to compare (2-10)
    - **comparison_type**: Type of comparison (seo, performance, content)
    """
    start_time = time.perf_counter()
    
    try:
        urls = [str(url) for url in request.urls]
        comparison_type = request.comparison_type or "seo"
        
        logger.info(f"Comparing {len(urls)} URLs for {comparison_type}")
        
        # Perform comparison
        comparison = await seo_service.compare_urls(urls, comparison_type)
        
        # Calculate processing time
        processing_time = time.perf_counter() - start_time
        
        # Add comparison information
        comparison_result = {
            'success': True,
            'urls': urls,
            'comparison_type': comparison_type,
            'processing_time': processing_time,
            'comparison': comparison,
            'timestamp': time.time()
        }
        
        logger.info(f"Comparison completed in {processing_time:.2f}s")
        
        return comparison_result
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check(
    factory = Depends(get_factory_dependency)
):
    """
    Get service health status.
    """
    try:
        # Get health status from factory
        health_status = factory.get_health_status()
        
        # Add additional health information
        health_response = {
            'status': health_status.get('status', 'unknown'),
            'service': 'SEO Service Ultra-Optimized',
            'version': '2.0.0',
            'environment': 'production',
            'uptime': time.time() - getattr(factory, '_start_time', time.time()),
            'components': health_status.get('dependencies', {})
        }
        
        return health_response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    factory = Depends(get_factory_dependency)
):
    """
    Get service performance metrics.
    """
    try:
        # Get performance stats from factory
        stats = factory.get_performance_stats()
        
        # Calculate metrics
        metrics = {
            'requests_total': stats.get('total_requests', 0),
            'requests_per_second': stats.get('requests_per_second', 0.0),
            'average_response_time': stats.get('average_response_time', 0.0),
            'error_rate': stats.get('error_rate', 0.0),
            'cache_hit_rate': stats.get('cache_hit_rate', 0.0),
            'memory_usage': stats.get('memory_usage', 0.0),
            'cpu_usage': stats.get('cpu_usage', 0.0)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics collection failed: {str(e)}"
        )

@router.get("/cache/status")
async def cache_status(
    factory = Depends(get_factory_dependency)
):
    """
    Get cache status and statistics.
    """
    try:
        cache = factory.get_cache()
        stats = cache.get_stats()
        
        return {
            'success': True,
            'cache_stats': stats,
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Cache status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Cache status check failed: {str(e)}"
        )

@router.post("/cache/clear")
async def clear_cache(
    category: Optional[str] = Query(None, description="Cache category to clear"),
    factory = Depends(get_factory_dependency)
):
    """
    Clear cache entries.
    
    - **category**: Optional category to clear (if not provided, clears all)
    """
    try:
        cache = factory.get_cache()
        
        if category:
            await cache.clear(category)
            logger.info(f"Cleared cache category: {category}")
        else:
            await cache.clear()
            logger.info("Cleared all cache")
        
        return {
            'success': True,
            'message': f"Cache cleared successfully",
            'category': category,
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Cache clear failed: {str(e)}"
        )

@router.get("/selenium/status")
async def selenium_status(
    factory = Depends(get_factory_dependency)
):
    """
    Get Selenium service status.
    """
    try:
        selenium_service = factory.get_selenium_service()
        status = selenium_service.get_status()
        
        return {
            'success': True,
            'selenium_status': status,
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Selenium status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Selenium status check failed: {str(e)}"
        )

@router.post("/selenium/restart")
async def restart_selenium(
    factory = Depends(get_factory_dependency)
):
    """
    Restart Selenium service.
    """
    try:
        selenium_service = factory.get_selenium_service()
        selenium_service.restart()
        
        logger.info("Selenium service restarted")
        
        return {
            'success': True,
            'message': "Selenium service restarted successfully",
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Selenium restart failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Selenium restart failed: {str(e)}"
        )

@router.get("/config")
async def get_config(
    factory = Depends(get_factory_dependency)
):
    """
    Get current service configuration.
    """
    try:
        # Get configuration from factory
        config = factory.config
        
        # Remove sensitive information
        safe_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                safe_config[key] = {}
                for sub_key, sub_value in value.items():
                    if 'key' in sub_key.lower() or 'password' in sub_key.lower() or 'secret' in sub_key.lower():
                        safe_config[key][sub_key] = '***HIDDEN***'
                    else:
                        safe_config[key][sub_key] = sub_value
            else:
                safe_config[key] = value
        
        return {
            'success': True,
            'config': safe_config,
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Config retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Config retrieval failed: {str(e)}"
        )

@router.post("/reload")
async def reload_service(
    factory = Depends(get_factory_dependency)
):
    """
    Reload service configuration and components.
    """
    try:
        # Reload factory
        factory.reload()
        
        logger.info("Service reloaded successfully")
        
        return {
            'success': True,
            'message': "Service reloaded successfully",
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Service reload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Service reload failed: {str(e)}"
        )

# Error handlers
@router.exception_handler(Exception)
async def global_exception_handler(request, exc) -> Any:
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            'success': False,
            'error': 'Internal server error',
            'message': str(exc) if getattr(request.app.state, 'debug', False) else 'Something went wrong',
            'timestamp': time.time()
        }
    )

# Middleware for request logging
@router.middleware("http")
async async def log_requests(request, call_next) -> Any:
    """Log all requests."""
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    processing_time = time.perf_counter() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {processing_time:.3f}s"
    )
    
    return response 