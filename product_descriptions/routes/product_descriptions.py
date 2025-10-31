from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio
import logging
from datetime import datetime
from ..dependencies.core import get_db_session, get_cache_manager, get_performance_monitor
from ..dependencies.auth import get_authenticated_user, require_permission
from ..routes.base import get_request_context, log_route_access
from ..schemas.base import BaseResponse, ErrorResponse
from ..pydantic_schemas import (
from ..api.enhanced_service import EnhancedProductDescriptionService
from ..async_database_api_operations import AsyncDatabaseManager
from typing import Any, List, Dict, Optional
"""
Product Descriptions Router

This module contains all routes related to product description generation,
management, and optimization. Uses clear dependency injection and
follows RESTful API patterns.
"""


# Import dependencies

# Import schemas
    ProductDescriptionRequest,
    ProductDescriptionResponse,
    ProductDescriptionUpdate,
    ProductDescriptionList,
    GenerationOptions,
    BatchGenerationRequest,
    BatchGenerationResponse
)

# Import services

# Initialize router
router = APIRouter(prefix="/product-descriptions", tags=["product-descriptions"])

# Logger
logger = logging.getLogger(__name__)

# Service instance
product_service = EnhancedProductDescriptionService()

# Route dependencies
async def get_product_service(
    context: Dict[str, Any] = Depends(get_request_context)
) -> EnhancedProductDescriptionService:
    """Get product description service with dependencies."""
    return product_service

async def get_db_manager(
    context: Dict[str, Any] = Depends(get_request_context)
) -> AsyncDatabaseManager:
    """Get database manager from context."""
    return context["async_io_manager"].db_manager

# Product Description Routes
@router.post("/generate", response_model=ProductDescriptionResponse)
async def generate_product_description(
    request: ProductDescriptionRequest,
    background_tasks: BackgroundTasks,
    context: Dict[str, Any] = Depends(get_request_context),
    service: EnhancedProductDescriptionService = Depends(get_product_service)
):
    """
    Generate a new product description.
    
    This endpoint generates product descriptions using AI models
    with caching, performance monitoring, and error handling.
    """
    try:
        # Log route access
        log_route_access(
            "generate_product_description",
            user_id=context["user"].id if context["user"] else None,
            product_name=request.product_name
        )
        
        # Start performance monitoring
        start_time = datetime.utcnow()
        context["performance_monitor"].start_operation("generate_description")
        
        # Check cache first
        cache_key = f"product_desc:{request.product_name}:{request.category}"
        cached_result = await context["cache_manager"].get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for product: {request.product_name}")
            context["performance_monitor"].end_operation("generate_description")
            return ProductDescriptionResponse(
                status="success",
                message="Description retrieved from cache",
                data=cached_result,
                cached=True
            )
        
        # Generate description
        result = await service.generate_description(
            product_name=request.product_name,
            category=request.category,
            features=request.features,
            target_audience=request.target_audience,
            tone=request.tone,
            length=request.length,
            language=request.language,
            options=request.options
        )
        
        # Cache the result
        await context["cache_manager"].set(
            cache_key,
            result,
            ttl=3600  # 1 hour
        )
        
        # End performance monitoring
        context["performance_monitor"].end_operation("generate_description")
        
        # Add background task for analytics
        background_tasks.add_task(
            service.log_generation_analytics,
            request.product_name,
            result.description_id,
            start_time
        )
        
        return ProductDescriptionResponse(
            status="success",
            message="Product description generated successfully",
            data=result,
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Error generating product description: {e}")
        context["error_monitor"].track_error("generate_description", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate product description"
        )

@router.get("/{description_id}", response_model=ProductDescriptionResponse)
async def get_product_description(
    description_id: str,
    context: Dict[str, Any] = Depends(get_request_context),
    service: EnhancedProductDescriptionService = Depends(get_product_service)
):
    """Get a specific product description by ID."""
    try:
        log_route_access("get_product_description", description_id=description_id)
        
        # Check cache first
        cache_key = f"description:{description_id}"
        cached_result = await context["cache_manager"].get(cache_key)
        
        if cached_result:
            return ProductDescriptionResponse(
                status="success",
                message="Description retrieved from cache",
                data=cached_result,
                cached=True
            )
        
        # Get from database
        result = await service.get_description(description_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product description not found"
            )
        
        # Cache the result
        await context["cache_manager"].set(cache_key, result, ttl=1800)
        
        return ProductDescriptionResponse(
            status="success",
            message="Product description retrieved successfully",
            data=result,
            cached=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving product description: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve product description"
        )

@router.get("/", response_model=ProductDescriptionList)
async def list_product_descriptions(
    page: int = 1,
    limit: int = 20,
    category: Optional[str] = None,
    user_id: Optional[str] = None,
    context: Dict[str, Any] = Depends(get_request_context),
    service: EnhancedProductDescriptionService = Depends(get_product_service)
):
    """List product descriptions with pagination and filtering."""
    try:
        log_route_access("list_product_descriptions", page=page, limit=limit)
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if limit < 1 or limit > 100:
            limit = 20
        
        # Get descriptions
        result = await service.list_descriptions(
            page=page,
            limit=limit,
            category=category,
            user_id=user_id or (context["user"].id if context["user"] else None)
        )
        
        return ProductDescriptionList(
            status="success",
            message="Product descriptions retrieved successfully",
            data=result.descriptions,
            pagination=result.pagination
        )
        
    except Exception as e:
        logger.error(f"Error listing product descriptions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve product descriptions"
        )

@router.put("/{description_id}", response_model=ProductDescriptionResponse)
async def update_product_description(
    description_id: str,
    update_data: ProductDescriptionUpdate,
    context: Dict[str, Any] = Depends(get_request_context),
    service: EnhancedProductDescriptionService = Depends(get_product_service)
):
    """Update a product description."""
    try:
        log_route_access("update_product_description", description_id=description_id)
        
        # Check if user owns the description or is admin
        description = await service.get_description(description_id)
        if not description:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product description not found"
            )
        
        if (context["user"] and 
            not context["user"].is_admin and 
            description.user_id != context["user"].id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this description"
            )
        
        # Update description
        result = await service.update_description(description_id, update_data)
        
        # Invalidate cache
        cache_key = f"description:{description_id}"
        await context["cache_manager"].delete(cache_key)
        
        return ProductDescriptionResponse(
            status="success",
            message="Product description updated successfully",
            data=result,
            cached=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating product description: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update product description"
        )

@router.delete("/{description_id}", response_model=BaseResponse)
async def delete_product_description(
    description_id: str,
    context: Dict[str, Any] = Depends(get_request_context),
    service: EnhancedProductDescriptionService = Depends(get_product_service)
):
    """Delete a product description."""
    try:
        log_route_access("delete_product_description", description_id=description_id)
        
        # Check if user owns the description or is admin
        description = await service.get_description(description_id)
        if not description:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product description not found"
            )
        
        if (context["user"] and 
            not context["user"].is_admin and 
            description.user_id != context["user"].id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this description"
            )
        
        # Delete description
        await service.delete_description(description_id)
        
        # Invalidate cache
        cache_key = f"description:{description_id}"
        await context["cache_manager"].delete(cache_key)
        
        return BaseResponse(
            status="success",
            message="Product description deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting product description: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete product description"
        )

# Batch Operations
@router.post("/batch/generate", response_model=BatchGenerationResponse)
async def batch_generate_descriptions(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    context: Dict[str, Any] = Depends(get_request_context),
    service: EnhancedProductDescriptionService = Depends(get_product_service)
):
    """Generate multiple product descriptions in batch."""
    try:
        log_route_access("batch_generate_descriptions", count=len(request.products))
        
        # Start performance monitoring
        context["performance_monitor"].start_operation("batch_generate")
        
        # Process batch
        results = await service.batch_generate_descriptions(
            products=request.products,
            options=request.options
        )
        
        # End performance monitoring
        context["performance_monitor"].end_operation("batch_generate")
        
        # Add background task for analytics
        background_tasks.add_task(
            service.log_batch_analytics,
            len(request.products),
            len(results.successful),
            len(results.failed)
        )
        
        return BatchGenerationResponse(
            status="success",
            message=f"Batch generation completed: {len(results.successful)} successful, {len(results.failed)} failed",
            data=results
        )
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch generation"
        )

# Streaming Generation
@router.post("/stream/generate")
async def stream_generate_description(
    request: ProductDescriptionRequest,
    context: Dict[str, Any] = Depends(get_request_context),
    service: EnhancedProductDescriptionService = Depends(get_product_service)
):
    """Stream product description generation in real-time."""
    try:
        log_route_access("stream_generate_description", product_name=request.product_name)
        
        async def generate_stream():
            
    """generate_stream function."""
async for chunk in service.stream_generate_description(
                product_name=request.product_name,
                category=request.category,
                features=request.features,
                target_audience=request.target_audience,
                tone=request.tone,
                length=request.length,
                language=request.language,
                options=request.options
            ):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Error in streaming generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start streaming generation"
        )

# Analytics and Metrics
@router.get("/analytics/summary")
async def get_analytics_summary(
    context: Dict[str, Any] = Depends(get_request_context),
    service: EnhancedProductDescriptionService = Depends(get_product_service)
):
    """Get analytics summary for product descriptions."""
    try:
        log_route_access("get_analytics_summary")
        
        summary = await service.get_analytics_summary()
        
        return {
            "status": "success",
            "message": "Analytics summary retrieved successfully",
            "data": summary
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics summary"
        )

# Health and Status
@router.get("/health/status")
async def get_service_health(
    context: Dict[str, Any] = Depends(get_request_context),
    service: EnhancedProductDescriptionService = Depends(get_product_service)
):
    """Get service health status."""
    try:
        health_status = await service.get_health_status()
        
        return {
            "status": "success",
            "message": "Service health status retrieved",
            "data": health_status
        }
        
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve health status"
        ) 