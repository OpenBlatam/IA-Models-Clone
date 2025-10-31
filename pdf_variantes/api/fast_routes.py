"""
PDF Variantes API - Fast Routes
Optimized routes for maximum performance
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import (
    APIRouter, Depends, HTTPException, Query, Path, Request, Response
)
from fastapi.responses import Response as FastAPIResponse

from ..models import PDFDocument
from ..utils.auth import get_current_user
from .dependencies import get_services, get_pdf_service
from .responses import create_success_response, create_paginated_response
from .performance import json_response, async_cache, PerformanceMonitor
from .validators import PaginationValidator

logger = logging.getLogger(__name__)

# Fast Router with performance optimizations
fast_router = APIRouter(prefix="/api/v1", tags=["Fast"])


@fast_router.get(
    "/pdf/documents/fast",
    summary="Fast document listing (optimized)",
    description="High-performance document listing with caching and compression"
)
@async_cache(ttl=60)  # Cache for 60 seconds
async def list_documents_fast(
    request: Request,
    user_id: str = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    services: Dict[str, Any] = Depends(get_services),
    compress: bool = Query(True, description="Compress response")
):
    """Ultra-fast document listing with caching"""
    try:
        pdf_service = get_pdf_service(services)
        
        # Validate pagination
        limit, offset = PaginationValidator.validate_pagination(limit, offset)
        
        # Get documents
        documents = await pdf_service.list_documents(user_id, limit, offset)
        
        # Create optimized response
        response_data = {
            "success": True,
            "data": documents,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', None)
        }
        
        # Use fast JSON response with optional compression
        return json_response(
            content=response_data,
            compress=compress,
            headers={
                "X-Total-Count": str(len(documents)),
                "Cache-Control": "public, max-age=60",
                "X-Cache-Status": "HIT"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fast document listing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@fast_router.get(
    "/pdf/documents/{document_id}/fast",
    summary="Fast document retrieval",
    description="High-performance single document retrieval"
)
@async_cache(ttl=300)  # Cache for 5 minutes
async def get_document_fast(
    request: Request,
    document_id: str = Path(..., description="Document ID"),
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services),
    compress: bool = Query(True)
):
    """Ultra-fast document retrieval"""
    try:
        pdf_service = get_pdf_service(services)
        
        document = await pdf_service.get_document(document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        response_data = {
            "success": True,
            "data": document,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', None)
        }
        
        return json_response(
            content=response_data,
            compress=compress,
            headers={
                "Cache-Control": "public, max-age=300",
                "ETag": f'"{document_id}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fast document retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@fast_router.get(
    "/health/fast",
    summary="Ultra-fast health check",
    description="Lightweight health check endpoint"
)
async def health_fast():
    """Fast health check - minimal processing"""
    return json_response(
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        },
        compress=False,
        headers={
            "Cache-Control": "public, max-age=10"
        }
    )






