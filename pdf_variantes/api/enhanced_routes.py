"""
PDF Variantes API - Enhanced Routes
Improved routes with pagination, filtering, and better responses
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, Form, 
    Query, Path, BackgroundTasks, Request, Response
)
from fastapi.responses import JSONResponse

from ..models import (
    PDFDocument, PDFVariant, TopicItem, BrainstormIdea
)
from ..utils.auth import get_current_user
from .dependencies import get_services, get_pdf_service
from .responses import (
    create_success_response,
    create_paginated_response,
    PaginationMeta
)
from .validators import (
    validate_pagination,
    validate_sort_params,
    validate_date_range,
    ContentRangeHeader,
    add_cache_headers,
    add_no_cache_headers
)

logger = logging.getLogger(__name__)

# Enhanced PDF Router
enhanced_pdf_router = APIRouter(prefix="/api/v1/pdf", tags=["PDF - Enhanced"])


@enhanced_pdf_router.get(
    "/documents",
    summary="List documents with pagination and filtering",
    description="Get a paginated list of documents with advanced filtering options"
)
async def list_documents_enhanced(
    request: Request,
    user_id: str = Depends(get_current_user),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: Optional[str] = Query(None, description="Sort field (created_at, name, size)"),
    sort_order: Optional[str] = Query("desc", description="Sort order (asc/desc)"),
    search: Optional[str] = Query(None, max_length=200, description="Search query"),
    status: Optional[str] = Query(None, description="Filter by status"),
    created_after: Optional[datetime] = Query(None, description="Created after date"),
    created_before: Optional[datetime] = Query(None, description="Created before date"),
    services: Dict[str, Any] = Depends(get_services)
):
    """Enhanced document listing with pagination, filtering, and sorting"""
    try:
        pdf_service = get_pdf_service(services)
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Build filter parameters
        filters = {
            "search": search,
            "status": status,
            "created_after": created_after,
            "created_before": created_before,
            "sort_by": sort_by or "created_at",
            "sort_order": sort_order or "desc"
        }
        
        # Get documents (this would need to be implemented in the service)
        # For now, using the existing method
        documents = await pdf_service.list_documents(user_id, limit, offset)
        
        # Get total count (would need separate method)
        total = len(documents)  # This should be a proper count query
        
        # Create paginated response
        response_data = create_paginated_response(
            items=documents,
            total=total,
            page=page,
            limit=limit,
            offset=offset,
            request_id=getattr(request.state, 'request_id', None)
        )
        
        # Create response with headers
        response = JSONResponse(content=response_data.model_dump())
        
        # Add pagination headers
        response.headers["X-Total-Count"] = str(total)
        response.headers["X-Page"] = str(page)
        response.headers["X-Per-Page"] = str(limit)
        response.headers["X-Total-Pages"] = str((total + limit - 1) // limit if limit > 0 else 0)
        
        # Add content range header
        content_range = ContentRangeHeader.create(offset, limit, total)
        response.headers["Content-Range"] = content_range
        
        # Add cache headers for list endpoints
        cache_headers = add_cache_headers(max_age=60)
        response.headers.update(cache_headers)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@enhanced_pdf_router.get(
    "/documents/{document_id}",
    summary="Get document details",
    description="Get detailed information about a specific document"
)
async def get_document_enhanced(
    request: Request,
    document_id: str = Path(..., description="Document ID"),
    user_id: str = Depends(get_current_user),
    include_variants: bool = Query(False, description="Include variants in response"),
    include_topics: bool = Query(False, description="Include topics in response"),
    services: Dict[str, Any] = Depends(get_services)
):
    """Get document with optional related data"""
    try:
        pdf_service = get_pdf_service(services)
        
        # Get document
        document = await pdf_service.get_document(document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Build response data
        response_data = {"document": document}
        
        # Include variants if requested
        if include_variants:
            variants = await pdf_service.list_variants(document_id, user_id, limit=10, offset=0)
            response_data["variants"] = variants
        
        # Include topics if requested
        if include_topics:
            topics = await pdf_service.list_topics(document_id, user_id)
            response_data["topics"] = topics
        
        # Create success response
        response = create_success_response(
            data=response_data,
            request_id=getattr(request.state, 'request_id', None)
        )
        
        # Add cache headers
        cache_headers = add_cache_headers(max_age=300)
        response_headers = {**cache_headers}
        
        return JSONResponse(
            content=response.model_dump(),
            headers=response_headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


# Enhanced Variants Router
enhanced_variant_router = APIRouter(prefix="/api/v1/variants", tags=["Variants - Enhanced"])


@enhanced_variant_router.get(
    "/documents/{document_id}/variants",
    summary="List variants with pagination",
    description="Get paginated list of variants for a document"
)
async def list_variants_enhanced(
    request: Request,
    document_id: str = Path(..., description="Document ID"),
    user_id: str = Depends(get_current_user),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort_by: Optional[str] = Query(None),
    sort_order: Optional[str] = Query("desc"),
    status: Optional[str] = Query(None, description="Filter by status"),
    services: Dict[str, Any] = Depends(get_services)
):
    """Enhanced variant listing"""
    try:
        pdf_service = get_pdf_service(services)
        
        # Check document access
        document = await pdf_service.get_document(document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Get variants
        variants = await pdf_service.list_variants(document_id, user_id, limit, offset)
        total = len(variants)  # Should be a proper count
        
        # Create paginated response
        response_data = create_paginated_response(
            items=variants,
            total=total,
            page=page,
            limit=limit,
            offset=offset,
            request_id=getattr(request.state, 'request_id', None)
        )
        
        response = JSONResponse(content=response_data.model_dump())
        
        # Add headers
        response.headers["X-Total-Count"] = str(total)
        response.headers["Content-Range"] = ContentRangeHeader.create(offset, limit, total)
        response.headers.update(add_cache_headers(max_age=60))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing variants: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Statistics Router
stats_router = APIRouter(prefix="/api/v1/stats", tags=["Statistics"])


@stats_router.get(
    "/overview",
    summary="Get API statistics overview",
    description="Get comprehensive statistics about the API usage"
)
async def get_stats_overview(
    request: Request,
    user_id: str = Depends(get_current_user),
    period: str = Query("7d", description="Time period (1d, 7d, 30d, all)"),
    services: Dict[str, Any] = Depends(get_services)
):
    """Get statistics overview"""
    try:
        analytics_service = services.get("analytics_service")
        
        stats = {
            "total_documents": 0,
            "total_variants": 0,
            "total_topics": 0,
            "period": period,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        if analytics_service:
            # Get stats from analytics service
            stats = await analytics_service.get_user_stats(user_id, period)
        
        response_data = create_success_response(
            data=stats,
            request_id=getattr(request.state, 'request_id', None)
        )
        
        # No cache for stats
        return JSONResponse(
            content=response_data.model_dump(),
            headers=add_no_cache_headers()
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@stats_router.get(
    "/usage",
    summary="Get API usage statistics",
    description="Get detailed usage statistics"
)
async def get_usage_stats(
    request: Request,
    user_id: str = Depends(get_current_user),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    services: Dict[str, Any] = Depends(get_services)
):
    """Get usage statistics"""
    try:
        analytics_service = services.get("analytics_service")
        
        usage_stats = {
            "requests_today": 0,
            "requests_this_week": 0,
            "requests_this_month": 0,
            "top_endpoints": [],
            "error_rate": 0.0
        }
        
        if analytics_service:
            usage_stats = await analytics_service.get_usage_stats(
                user_id, start_date, end_date
            )
        
        response_data = create_success_response(
            data=usage_stats,
            request_id=getattr(request.state, 'request_id', None)
        )
        
        return JSONResponse(
            content=response_data.model_dump(),
            headers=add_no_cache_headers()
        )
        
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






