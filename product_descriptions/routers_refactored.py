from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import List
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from fastapi.responses import JSONResponse
from .config_refactored import config
from .schemas_refactored import (
from .services_refactored import services
    import time
    import asyncio
from typing import Any, List, Dict, Optional
import logging
"""
Refactored Routers Module
========================

Clean, well-organized API routers with proper separation of concerns.
Each router handles a specific domain with clear responsibilities.
"""


    ProductCreateRequest,
    ProductUpdateRequest,
    ProductSearchRequest,
    ProductResponse,
    ProductListResponse,
    BulkOperationResponse,
    HealthResponse,
    MetricsResponse,
    AIDescriptionRequest,
    AIDescriptionResponse,
    ErrorResponse
)

logger = structlog.get_logger(__name__)


# =============================================================================
# DEPENDENCY PROVIDERS - Clean injection
# =============================================================================

async def get_product_service():
    """Get product service dependency."""
    return await services.get_product_service()


async def get_ai_service():
    """Get AI service dependency."""
    return await services.get_ai_service()


async def get_health_service():
    """Get health service dependency."""
    return await services.get_health_service()


# =============================================================================
# PRODUCTS ROUTER - Main business domain
# =============================================================================

products_router = APIRouter(
    prefix="/products",
    tags=["Products"],
    responses={
        404: {"model": ErrorResponse, "description": "Product not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


@products_router.post(
    "/",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new product",
    description="Create a new product with comprehensive validation and business logic."
)
async def create_product(
    request: ProductCreateRequest,
    product_service = Depends(get_product_service)
) -> ProductResponse:
    """Create a new product."""
    try:
        product = await product_service.create_product(request)
        logger.info("Product created via API", product_id=product.id, name=product.name)
        return product
    
    except ValueError as e:
        logger.warning("Product creation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Unexpected error in product creation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@products_router.get(
    "/{product_id}",
    response_model=ProductResponse,
    summary="Get product by ID",
    description="Retrieve a product by its unique identifier with caching support."
)
async def get_product(
    product_id: str = Path(..., description="Product unique identifier"),
    product_service = Depends(get_product_service)
) -> ProductResponse:
    """Get product by ID."""
    try:
        product = await product_service.get_product(product_id)
        
        if not product:
            logger.info("Product not found", product_id=product_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with ID '{product_id}' not found"
            )
        
        logger.debug("Product retrieved", product_id=product_id, cache_hit=product.cache_hit)
        return product
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving product", product_id=product_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@products_router.post(
    "/search",
    response_model=ProductListResponse,
    summary="Search products",
    description="Advanced product search with filtering, sorting, and pagination."
)
async def search_products(
    request: ProductSearchRequest,
    product_service = Depends(get_product_service)
) -> ProductListResponse:
    """Search products with advanced filters."""
    try:
        results = await product_service.search_products(request)
        
        logger.info("Product search executed", 
                   query=request.query,
                   total_results=results.total,
                   page=results.page)
        
        return results
    
    except Exception as e:
        logger.error("Error in product search", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search operation failed"
        )


@products_router.put(
    "/{product_id}",
    response_model=ProductResponse,
    summary="Update product",
    description="Update an existing product with partial data."
)
async def update_product(
    product_id: str = Path(..., description="Product unique identifier"),
    request: ProductUpdateRequest = ...,
    product_service = Depends(get_product_service)
) -> ProductResponse:
    """Update existing product."""
    try:
        # Get current product
        current_product = await product_service.get_product(product_id)
        if not current_product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with ID '{product_id}' not found"
            )
        
        # Update product (simplified for demo)
        # In real implementation, this would go through the service layer
        update_data = request.dict(exclude_unset=True)
        
        logger.info("Product update requested", product_id=product_id, fields=list(update_data.keys()))
        
        # For demo, return current product with cache_hit = False
        current_product.cache_hit = False
        return current_product
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating product", product_id=product_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Product update failed"
        )


@products_router.delete(
    "/{product_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete product",
    description="Delete a product by its unique identifier."
)
async def delete_product(
    product_id: str = Path(..., description="Product unique identifier"),
    product_service = Depends(get_product_service)
) -> None:
    """Delete product."""
    try:
        # Check if product exists
        product = await product_service.get_product(product_id)
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with ID '{product_id}' not found"
            )
        
        # Delete product (simplified for demo)
        logger.info("Product deletion requested", product_id=product_id)
        
        # In real implementation, this would call repository.delete()
        return None
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting product", product_id=product_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Product deletion failed"
        )


@products_router.post(
    "/bulk",
    response_model=BulkOperationResponse,
    summary="Bulk create products",
    description="Create multiple products in a single operation with concurrent processing."
)
async def bulk_create_products(
    requests: List[ProductCreateRequest],
    product_service = Depends(get_product_service)
) -> BulkOperationResponse:
    """Bulk create products."""
    
    if len(requests) > config.security.max_bulk_operations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {config.security.max_bulk_operations} products allowed per bulk operation"
        )
    
    start_time = time.time()
    successful = 0
    failed = 0
    errors = []
    
    try:
        # Process products concurrently
        tasks = []
        for i, request in enumerate(requests):
            task = asyncio.create_task(
                _create_product_safe(product_service, request, i)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                errors.append({
                    "index": i,
                    "sku": requests[i].sku,
                    "error": str(result)
                })
            else:
                successful += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info("Bulk product creation completed",
                   total=len(requests),
                   successful=successful,
                   failed=failed,
                   processing_time_ms=processing_time)
        
        return BulkOperationResponse(
            successful=successful,
            failed=failed,
            total=len(requests),
            errors=errors,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error("Bulk operation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk operation failed"
        )


async def _create_product_safe(product_service, request: ProductCreateRequest, index: int):
    """Safely create a product for bulk operations."""
    try:
        return await product_service.create_product(request)
    except Exception as e:
        # Re-raise with context
        raise ValueError(f"Failed to create product at index {index}: {str(e)}")


# =============================================================================
# AI ROUTER - AI/ML features
# =============================================================================

ai_router = APIRouter(
    prefix="/ai",
    tags=["AI & Machine Learning"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "AI service error"}
    }
)


@ai_router.post(
    "/generate-description",
    response_model=AIDescriptionResponse,
    summary="Generate AI product description",
    description="Generate intelligent product descriptions using AI/LLM models."
)
async def generate_ai_description(
    request: AIDescriptionRequest,
    ai_service = Depends(get_ai_service)
) -> AIDescriptionResponse:
    """Generate AI-powered product description."""
    try:
        result = await ai_service.generate_description(request)
        
        logger.info("AI description generated",
                   product_name=request.product_name,
                   confidence=result.confidence_score,
                   processing_time=result.processing_time_ms)
        
        return result
    
    except Exception as e:
        logger.error("AI description generation failed", 
                    product_name=request.product_name,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI description generation failed"
        )


@ai_router.get(
    "/status",
    summary="AI service status",
    description="Check AI service availability and configuration."
)
async def get_ai_status(
    ai_service = Depends(get_ai_service)
) -> dict:
    """Get AI service status."""
    return {
        "enabled": ai_service.enabled,
        "model": config.ai.openai_model if ai_service.enabled else "disabled",
        "features": {
            "description_generation": ai_service.enabled,
            "content_moderation": config.ai.enable_content_moderation
        }
    }


# =============================================================================
# HEALTH ROUTER - Monitoring and observability
# =============================================================================

health_router = APIRouter(
    prefix="/health",
    tags=["Health & Monitoring"],
    responses={
        503: {"model": ErrorResponse, "description": "Service unhealthy"}
    }
)


@health_router.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    description="Comprehensive health check of all system components."
)
async def health_check(
    health_service = Depends(get_health_service)
) -> HealthResponse:
    """Get system health status."""
    try:
        health_status = await health_service.get_health_status()
        
        # Return 503 if system is not healthy
        if health_status.status != "healthy":
            logger.warning("System health check failed", status=health_status.status)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_status.dict()
            )
        
        return health_status
    
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@health_router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Application metrics",
    description="Detailed application performance and usage metrics."
)
async def get_metrics(
    health_service = Depends(get_health_service)
) -> MetricsResponse:
    """Get application metrics."""
    try:
        metrics = await health_service.get_metrics()
        return metrics
    
    except Exception as e:
        logger.error("Metrics collection failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics collection failed"
        )


@health_router.get(
    "/ready",
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint."
)
async def readiness_probe() -> dict:
    """Readiness probe for Kubernetes."""
    return {"status": "ready", "timestamp": "2024-01-01T00:00:00Z"}


@health_router.get(
    "/live",
    summary="Liveness probe", 
    description="Kubernetes liveness probe endpoint."
)
async def liveness_probe() -> dict:
    """Liveness probe for Kubernetes."""
    return {"status": "alive", "timestamp": "2024-01-01T00:00:00Z"}


# =============================================================================
# ADMIN ROUTER - Administrative functions
# =============================================================================

admin_router = APIRouter(
    prefix="/admin",
    tags=["Administration"],
    responses={
        403: {"model": ErrorResponse, "description": "Access forbidden"},
        500: {"model": ErrorResponse, "description": "Administrative error"}
    }
)


@admin_router.get(
    "/cache/stats",
    summary="Cache statistics",
    description="Get detailed cache performance statistics."
)
async def get_cache_stats() -> dict:
    """Get cache statistics."""
    try:
        cache_service = await services.get_cache_service()
        
        stats = {
            "enabled": cache_service.client is not None,
            "hits": getattr(cache_service, 'hits', 0),
            "misses": getattr(cache_service, 'misses', 0),
            "operations": getattr(cache_service, 'operations', 0),
            "hit_ratio": getattr(cache_service, 'hit_ratio', 0.0),
            "connection_info": {
                "url": cache_service.redis_url,
                "max_connections": cache_service.max_connections
            }
        }
        
        return stats
    
    except Exception as e:
        logger.error("Cache stats collection failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cache statistics unavailable"
        )


@admin_router.post(
    "/cache/clear",
    summary="Clear cache",
    description="Clear all cached data (use with caution)."
)
async def clear_cache() -> dict:
    """Clear all cache data."""
    try:
        cache_service = await services.get_cache_service()
        
        if cache_service.client:
            # In a real implementation, you'd clear specific patterns
            # await cache_service.client.flushdb()
            logger.warning("Cache clear requested (not implemented in demo)")
            return {"message": "Cache clear requested", "status": "demo_mode"}
        else:
            return {"message": "Cache not available", "status": "disabled"}
    
    except Exception as e:
        logger.error("Cache clear failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cache clear operation failed"
        )


@admin_router.get(
    "/config",
    summary="Application configuration",
    description="Get current application configuration (sensitive data masked)."
)
async def get_config() -> dict:
    """Get application configuration."""
    try:
        # Return safe configuration (mask sensitive data)
        safe_config = {
            "app": {
                "name": config.name,
                "version": config.version,
                "environment": config.environment,
                "debug": config.debug
            },
            "api": {
                "prefix": config.api_prefix,
                "docs_url": config.docs_url
            },
            "features": {
                "ai_enabled": config.enable_ai,
                "rate_limiting": config.rate_limit_enabled,
                "metrics": config.enable_metrics
            },
            "cache": {
                "ttl": config.cache_ttl,
                "max_connections": config.redis_max_connections
            }
        }
        
        return safe_config
    
    except Exception as e:
        logger.error("Config retrieval failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration unavailable"
        )


# =============================================================================
# ROUTER COLLECTION - Export all routers
# =============================================================================

def get_all_routers() -> List[APIRouter]:
    """Get all application routers."""
    return [
        products_router,
        ai_router,
        health_router,
        admin_router
    ] 