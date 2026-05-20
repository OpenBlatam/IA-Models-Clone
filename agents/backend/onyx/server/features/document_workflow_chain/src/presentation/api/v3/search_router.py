"""
Search Router
=============

FastAPI router for search operations and search management.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from pydantic import BaseModel, Field

from ...shared.services.search_service import (
    SearchType,
    SearchOperator,
    SearchBackend,
    search_documents,
    index_document,
    delete_document,
    get_search_suggestions,
    get_search_facets,
    get_search_stats
)
from ...shared.middleware.auth import get_current_user_optional
from ...shared.middleware.rate_limiter import rate_limit
from ...shared.middleware.metrics_middleware import record_search_metrics
from ...shared.utils.decorators import log_execution, measure_performance


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/search", tags=["Search Operations"])


# Request/Response models
class SearchRequest(BaseModel):
    """Search request"""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    search_type: SearchType = Field(SearchType.FULL_TEXT, description="Type of search")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    facets: Optional[List[str]] = Field(None, description="Facet fields")
    sort: Optional[List[Dict[str, str]]] = Field(None, description="Sort criteria")
    limit: int = Field(100, description="Maximum number of results", ge=1, le=1000)
    offset: int = Field(0, description="Number of results to skip", ge=0)
    operator: SearchOperator = Field(SearchOperator.AND, description="Search operator")
    fuzzy: bool = Field(False, description="Enable fuzzy search")
    highlight: bool = Field(True, description="Enable highlighting")


class SearchResultResponse(BaseModel):
    """Search result response"""
    id: str
    title: str
    content: str
    score: float
    highlights: List[str]
    metadata: Dict[str, Any]
    facets: Dict[str, List[Dict[str, Any]]]


class SearchResponse(BaseModel):
    """Search response"""
    results: List[SearchResultResponse]
    total: int
    query: str
    search_type: str
    took: float
    facets: Dict[str, List[Dict[str, Any]]]


class DocumentIndexRequest(BaseModel):
    """Document index request"""
    document_id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    tags: Optional[List[str]] = Field(None, description="Document tags")


class SearchSuggestionsResponse(BaseModel):
    """Search suggestions response"""
    suggestions: List[str]
    query: str
    total: int


class SearchFacetsResponse(BaseModel):
    """Search facets response"""
    facets: Dict[str, List[Dict[str, Any]]]
    query: str
    total: int


class SearchStatsResponse(BaseModel):
    """Search statistics response"""
    backend: str
    total_documents: int
    total_size: int
    indices: int
    timestamp: str


# Search operations endpoints
@router.post("/search", response_model=SearchResponse)
@rate_limit(requests=30, window=60)  # 30 requests per minute
@record_search_metrics
@log_execution
@measure_performance
async def search_documents_endpoint(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> SearchResponse:
    """Search documents"""
    try:
        # Convert sort criteria
        sort_criteria = []
        if request.sort:
            for sort_item in request.sort:
                for field, order in sort_item.items():
                    sort_criteria.append((field, order))
        
        # Perform search
        results = await search_documents(
            query=request.query,
            search_type=request.search_type,
            limit=request.limit
        )
        
        # Convert results to response format
        result_responses = []
        for result in results:
            # Convert facets to the expected format
            facets_dict = {}
            for field, facet_list in result.facets.items():
                facets_dict[field] = [
                    {"value": value, "count": count}
                    for value, count in facet_list
                ]
            
            result_responses.append(SearchResultResponse(
                id=result.id,
                title=result.title,
                content=result.content,
                score=result.score,
                highlights=result.highlights,
                metadata=result.metadata,
                facets=facets_dict
            ))
        
        # Get facets if requested
        facets_response = {}
        if request.facets:
            facets = await get_search_facets("default", request.query, request.facets)
            for field, facet_list in facets.items():
                facets_response[field] = [
                    {"value": value, "count": count}
                    for value, count in facet_list
                ]
        
        return SearchResponse(
            results=result_responses,
            total=len(result_responses),
            query=request.query,
            search_type=request.search_type.value,
            took=0.0,  # In real implementation, measure actual time
            facets=facets_response
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search", response_model=SearchResponse)
@rate_limit(requests=30, window=60)  # 30 requests per minute
@record_search_metrics
@log_execution
@measure_performance
async def search_documents_get_endpoint(
    q: str = Query(..., description="Search query", min_length=1, max_length=1000),
    search_type: SearchType = Query(SearchType.FULL_TEXT, description="Type of search"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    facets: Optional[str] = Query(None, description="Facet fields (comma-separated)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> SearchResponse:
    """Search documents (GET endpoint)"""
    try:
        # Parse facets
        facet_fields = facets.split(",") if facets else []
        
        # Perform search
        results = await search_documents(
            query=q,
            search_type=search_type,
            limit=limit
        )
        
        # Convert results to response format
        result_responses = []
        for result in results:
            # Convert facets to the expected format
            facets_dict = {}
            for field, facet_list in result.facets.items():
                facets_dict[field] = [
                    {"value": value, "count": count}
                    for value, count in facet_list
                ]
            
            result_responses.append(SearchResultResponse(
                id=result.id,
                title=result.title,
                content=result.content,
                score=result.score,
                highlights=result.highlights,
                metadata=result.metadata,
                facets=facets_dict
            ))
        
        # Get facets if requested
        facets_response = {}
        if facet_fields:
            facets = await get_search_facets("default", q, facet_fields)
            for field, facet_list in facets.items():
                facets_response[field] = [
                    {"value": value, "count": count}
                    for value, count in facet_list
                ]
        
        return SearchResponse(
            results=result_responses,
            total=len(result_responses),
            query=q,
            search_type=search_type.value,
            took=0.0,  # In real implementation, measure actual time
            facets=facets_response
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/index")
@rate_limit(requests=20, window=60)  # 20 requests per minute
@record_search_metrics
@log_execution
@measure_performance
async def index_document_endpoint(
    request: DocumentIndexRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, str]:
    """Index document for search"""
    try:
        # Prepare document
        document = {
            "title": request.title,
            "content": request.content,
            "tags": request.tags or []
        }
        
        # Add user metadata
        metadata = request.metadata or {}
        if current_user:
            metadata["indexed_by"] = current_user.get("id")
            metadata["indexed_by_email"] = current_user.get("email")
        
        # Index document
        success = await index_document(
            index_name="default",
            document_id=request.document_id,
            document=document,
            metadata=metadata
        )
        
        if success:
            return {"message": f"Document {request.document_id} indexed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to index document")
    
    except Exception as e:
        logger.error(f"Document indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {str(e)}")


@router.delete("/index/{document_id}")
@rate_limit(requests=20, window=60)  # 20 requests per minute
@record_search_metrics
@log_execution
@measure_performance
async def delete_document_endpoint(
    document_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, str]:
    """Delete document from search index"""
    try:
        # Delete document
        success = await delete_document("default", document_id)
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
    
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document deletion failed: {str(e)}")


@router.get("/suggestions", response_model=SearchSuggestionsResponse)
@rate_limit(requests=50, window=60)  # 50 requests per minute
@record_search_metrics
@log_execution
@measure_performance
async def get_search_suggestions_endpoint(
    q: str = Query(..., description="Search query for suggestions", min_length=1),
    limit: int = Query(10, description="Maximum number of suggestions", ge=1, le=50),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> SearchSuggestionsResponse:
    """Get search suggestions"""
    try:
        # Get suggestions
        suggestions = await get_search_suggestions(q, limit)
        
        return SearchSuggestionsResponse(
            suggestions=suggestions,
            query=q,
            total=len(suggestions)
        )
    
    except Exception as e:
        logger.error(f"Failed to get search suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get search suggestions: {str(e)}")


@router.get("/facets", response_model=SearchFacetsResponse)
@rate_limit(requests=30, window=60)  # 30 requests per minute
@record_search_metrics
@log_execution
@measure_performance
async def get_search_facets_endpoint(
    q: str = Query(..., description="Search query", min_length=1),
    fields: str = Query(..., description="Facet fields (comma-separated)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> SearchFacetsResponse:
    """Get search facets"""
    try:
        # Parse facet fields
        facet_fields = fields.split(",")
        
        # Get facets
        facets = await get_search_facets("default", q, facet_fields)
        
        # Convert facets to the expected format
        facets_response = {}
        for field, facet_list in facets.items():
            facets_response[field] = [
                {"value": value, "count": count}
                for value, count in facet_list
            ]
        
        return SearchFacetsResponse(
            facets=facets_response,
            query=q,
            total=sum(len(facet_list) for facet_list in facets.values())
        )
    
    except Exception as e:
        logger.error(f"Failed to get search facets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get search facets: {str(e)}")


@router.get("/stats", response_model=SearchStatsResponse)
@rate_limit(requests=10, window=60)  # 10 requests per minute
@record_search_metrics
@log_execution
@measure_performance
async def get_search_stats_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> SearchStatsResponse:
    """Get search statistics"""
    try:
        # Get search stats
        stats = await get_search_stats()
        
        return SearchStatsResponse(
            backend=stats.get("backend", "unknown"),
            total_documents=stats.get("total_documents", 0),
            total_size=stats.get("total_size", 0),
            indices=stats.get("indices", 0),
            timestamp=stats.get("timestamp", "2024-01-01T00:00:00Z")
        )
    
    except Exception as e:
        logger.error(f"Failed to get search stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get search stats: {str(e)}")


# Health check endpoint
@router.get("/health")
@log_execution
async def search_service_health_check() -> Dict[str, Any]:
    """Search service health check"""
    try:
        # Check if search service is running
        stats = await get_search_stats()
        
        return {
            "status": "healthy",
            "backend": stats.get("backend", "unknown"),
            "total_documents": stats.get("total_documents", 0),
            "indices": stats.get("indices", 0),
            "timestamp": "2024-01-01T00:00:00Z"  # In real implementation, use actual timestamp
        }
    
    except Exception as e:
        logger.error(f"Search service health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"  # In real implementation, use actual timestamp
        }


