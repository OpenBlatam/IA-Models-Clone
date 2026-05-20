"""
Ultra-Fast Routes
Highly optimized routes for maximum performance
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import (
    APIRouter, Depends, HTTPException, Query, Path, Request
)
from fastapi.responses import Response
import asyncio

from ..utils.auth import get_current_user
from ..dependencies import get_services
from .query_cache import cache_query
from .async_pool import ParallelExecutor
from .response_optimizer import ResponseOptimizer, optimize_response
from .lazy_loading import PrefetchOptimizer

logger = logging.getLogger(__name__)

# Ultra-fast router
ultra_fast_router = APIRouter(prefix="/api/v1", tags=["Ultra-Fast"])

# Prefetch optimizer
prefetch_optimizer = PrefetchOptimizer()


@ultra_fast_router.get(
    "/pdf/documents/ultra-fast",
    summary="Ultra-fast document listing",
    description="Maximum performance with aggressive caching and optimization"
)
@cache_query(ttl=30)
@optimize_response(compress=True, minify=True, cache=True, cache_ttl=60)
async def list_documents_ultra_fast(
    request: Request,
    user_id: str = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    services: Dict[str, Any] = Depends(get_services)
):
    """Ultra-fast document listing with all optimizations"""
    try:
        pdf_service = services.get("pdf_service")
        
        # Parallel execution of count and fetch
        count_task = pdf_service.count_documents(user_id) if hasattr(pdf_service, 'count_documents') else None
        docs_task = pdf_service.list_documents(user_id, limit, offset)
        
        if count_task:
            total, documents = await asyncio.gather(count_task, docs_task)
        else:
            documents = await docs_task
            total = len(documents)
        
        # Optimize response
        response_data = {
            "success": True,
            "data": documents,
            "total": total,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Return optimized response
        return ResponseOptimizer.create_optimized_response(
            response_data,
            compress=True,
            headers={
                "X-Total-Count": str(total),
                "Cache-Control": "public, max-age=60"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in ultra-fast listing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@ultra_fast_router.get(
    "/pdf/documents/{document_id}/ultra-fast",
    summary="Ultra-fast document retrieval",
    description="Single document with aggressive caching"
)
@cache_query(ttl=300)  # Cache for 5 minutes
@optimize_response(compress=True, cache=True, cache_ttl=300)
async def get_document_ultra_fast(
    request: Request,
    document_id: str = Path(...),
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services),
    include_related: bool = Query(False, description="Include related data")
):
    """Ultra-fast document retrieval"""
    try:
        pdf_service = services.get("pdf_service")
        
        # Parallel fetch of document and related data if needed
        tasks = [pdf_service.get_document(document_id, user_id)]
        
        if include_related:
            # Fetch variants and topics in parallel
            tasks.extend([
                pdf_service.list_variants(document_id, user_id, limit=10, offset=0),
                pdf_service.list_topics(document_id, user_id)
            ])
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        document = results[0]
        if not document or isinstance(document, Exception):
            raise HTTPException(status_code=404, detail="Document not found")
        
        response_data = {
            "success": True,
            "data": {"document": document}
        }
        
        if include_related:
            response_data["data"]["variants"] = results[1] if len(results) > 1 else []
            response_data["data"]["topics"] = results[2] if len(results) > 2 else []
        
        return ResponseOptimizer.create_optimized_response(
            response_data,
            compress=True,
            headers={
                "Cache-Control": "public, max-age=300",
                "ETag": f'"{document_id}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ultra-fast retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@ultra_fast_router.get(
    "/batch/ultra-fast",
    summary="Batch operations ultra-fast",
    description="Process multiple operations in parallel"
)
async def batch_ultra_fast(
    request: Request,
    operation: str = Query(..., description="Operation name"),
    item_ids: List[str] = Query(..., description="Item IDs"),
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services),
    batch_size: int = Query(50, ge=1, le=200)
):
    """Batch operations with parallel processing"""
    try:
        pdf_service = services.get("pdf_service")
        
        # Process in parallel batches
        executor = ParallelExecutor()
        
        async def process_item(item_id: str):
            # Operation-specific logic
            if operation == "get_documents":
                return await pdf_service.get_document(item_id, user_id)
            return None
        
        # Execute in batches
        results = await executor.execute_batch(
            item_ids,
            process_item,
            batch_size=batch_size,
            parallel=True
        )
        
        response_data = {
            "success": True,
            "data": {
                "results": results,
                "total": len(results),
                "processed": len([r for r in results if r is not None])
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return ResponseOptimizer.create_optimized_response(
            response_data,
            compress=True
        )
        
    except Exception as e:
        logger.error(f"Error in batch operation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

