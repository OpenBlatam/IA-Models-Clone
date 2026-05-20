"""
Perplexity Routes - Routes for Perplexity-style query processing
=================================================================

Endpoints for processing queries in Perplexity style with search results
and formatted responses.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from ...core.perplexity import (
    PerplexityProcessor,
    PerplexityService,
    ProcessedQuery,
    QueryType,
    QueryProcessingError,
    LLMProviderError
)
from ..utils import handle_route_errors
from ..serializers import serialize_search_results

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/perplexity", tags=["perplexity"])


class SearchResultInput(BaseModel):
    """Input model for a search result."""
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the search result")
    snippet: str = Field(..., description="Snippet or excerpt from the result")
    content: Optional[str] = Field(None, description="Full content if available")
    source: Optional[str] = Field(None, description="Source name")
    timestamp: Optional[str] = Field(None, description="Timestamp in ISO format")


class QueryRequest(BaseModel):
    """Request model for a Perplexity-style query."""
    query: str = Field(..., description="The user's query")
    search_results: Optional[List[SearchResultInput]] = Field(
        default=None,
        description="Search results to use for answering the query"
    )
    include_metadata: bool = Field(
        default=False,
        description="Whether to include query processing metadata in response"
    )
    use_llm: bool = Field(
        default=False,
        description="Whether to use LLM for answer generation (requires LLM provider configured)"
    )


class QueryResponse(BaseModel):
    """Response model for a processed query."""
    query: str
    query_type: str
    answer: str
    metadata: Optional[Dict[str, Any]] = None


class ProcessedQueryResponse(BaseModel):
    """Response model for query processing (without answer generation)."""
    query: str
    query_type: str
    requires_citations: bool
    search_result_count: int
    metadata: Dict[str, Any]


# Initialize service and processor
_service: Optional[PerplexityService] = None
_processor: Optional[PerplexityProcessor] = None


def get_service() -> PerplexityService:
    """Get or create the Perplexity service instance."""
    global _service
    if _service is None:
        _service = PerplexityService()
    return _service


def get_processor() -> PerplexityProcessor:
    """Get or create the Perplexity processor instance."""
    global _processor
    if _processor is None:
        _processor = PerplexityProcessor()
    return _processor


@router.post("/query", response_model=QueryResponse)
@handle_route_errors("processing query")
async def process_query(
    request: QueryRequest,
    service: PerplexityService = Depends(get_service)
):
    """
    Process a query in Perplexity style and generate an answer.
    
    This endpoint processes the query, detects its type, and generates
    a formatted answer. If use_llm is True, it will attempt to use
    an LLM provider from the app state.
    
    Args:
        request: Query request with query string and optional search results
        service: Perplexity service instance
        
    Returns:
        Query response with formatted answer
        
    Raises:
        HTTPException: If processing fails
    """
    # Convert search results to dict format
    search_results_dict = serialize_search_results(request.search_results)
    
    # Process the query and generate answer
    result = await service.answer_query(
        query=request.query,
        search_results=search_results_dict,
        use_llm=request.use_llm
    )
    
    response_data = {
        'query': result.get('query', request.query),
        'query_type': result.get('query_type', 'general'),
        'answer': result.get('answer', '')
    }
    
    if request.include_metadata:
        response_data['metadata'] = result.get('metadata')
    
    return QueryResponse(**response_data)


@router.post("/process", response_model=ProcessedQueryResponse)
@handle_route_errors("processing query")
async def process_query_only(
    request: QueryRequest,
    processor: PerplexityProcessor = Depends(get_processor)
):
    """
    Process a query and return metadata without generating an answer.
    
    Useful for understanding how a query will be processed before
    generating the actual answer.
    
    Args:
        request: Query request
        processor: Perplexity processor instance
        
    Returns:
        Processed query metadata
    """
    # Convert search results to dict format
    search_results_dict = serialize_search_results(request.search_results)
    
    # Process the query
    processed = processor.process_query(
        query=request.query,
        search_results=search_results_dict
    )
    
    return ProcessedQueryResponse(
        query=processed.original_query,
        query_type=processed.query_type.value,
        requires_citations=processed.requires_citations,
        search_result_count=len(processed.search_results),
        metadata=processed.metadata or {}
    )


@router.post("/prompt")
@handle_route_errors("building prompt")
async def get_prompt(
    request: QueryRequest,
    processor: PerplexityProcessor = Depends(get_processor),
    include_planning: bool = True
):
    """
    Get the LLM prompt for a query without generating an answer.
    
    Useful for debugging or custom LLM integration.
    
    Args:
        request: Query request
        processor: Perplexity processor instance
        include_planning: Whether to include planning instructions in prompt
        
    Returns:
        The complete prompt that would be sent to the LLM
    """
    # Convert search results to dict format
    search_results_dict = serialize_search_results(request.search_results)
    
    # Process the query
    processed = processor.process_query(
        query=request.query,
        search_results=search_results_dict
    )
    
    # Build prompt
    prompt = processor.build_llm_prompt(processed, include_planning=include_planning)
    
    return {
        "query": processed.original_query,
        "query_type": processed.query_type.value,
        "prompt": prompt,
        "metadata": processed.metadata if request.include_metadata else None
    }


class ValidationRequest(BaseModel):
    """Request model for answer validation."""
    answer: str = Field(..., description="The answer text to validate")
    query_type: str = Field(default="general", description="The query type")


@router.post("/validate")
@handle_route_errors("validating answer")
async def validate_answer(
    request: ValidationRequest,
    service: PerplexityService = Depends(get_service)
):
    """
    Validate an answer against Perplexity formatting rules.
    
    Args:
        request: Validation request with answer and query type
        service: Perplexity service instance
        
    Returns:
        Validation results with issues and suggestions
    """
    return service.validate_answer(request.answer, request.query_type)


@router.post("/batch")
@handle_route_errors("batch processing queries")
async def batch_process(
    queries: List[QueryRequest],
    service: PerplexityService = Depends(get_service),
    use_llm: bool = False
):
    """
    Process multiple queries in batch.
    
    Args:
        queries: List of query requests
        service: Perplexity service instance
        use_llm: Whether to use LLM for all queries
        
    Returns:
        List of results for each query
    """
    query_data = [
        {
            'query': q.query,
            'search_results': serialize_search_results(q.search_results)
        }
        for q in queries
    ]
    
    results = await service.batch_process(query_data)
    return {"results": results, "total": len(results)}


@router.get("/cache/stats")
async def get_cache_stats(service: PerplexityService = Depends(get_service)):
    """Get cache statistics."""
    return service.get_cache_stats()


@router.post("/cache/clear")
async def clear_cache(service: PerplexityService = Depends(get_service)):
    """Clear the cache."""
    service.clear_cache()
    return {"message": "Cache cleared successfully"}


@router.get("/metrics")
async def get_metrics(service: PerplexityService = Depends(get_service)):
    """Get metrics statistics."""
    return service.get_metrics()


@router.post("/metrics/clear")
async def clear_metrics(service: PerplexityService = Depends(get_service)):
    """Clear metrics."""
    service.clear_metrics()
    return {"message": "Metrics cleared successfully"}


@router.get("/query-types")
async def get_query_types():
    """
    Get list of supported query types.
    
    Returns:
        List of query type names and descriptions
    """
    return {
        "query_types": [
            {
                "name": "academic_research",
                "description": "Academic research queries requiring detailed, scientific write-ups"
            },
            {
                "name": "recent_news",
                "description": "Recent news events with timestamps and sources"
            },
            {
                "name": "weather",
                "description": "Weather forecasts and climate information"
            },
            {
                "name": "people",
                "description": "Biographical information about people"
            },
            {
                "name": "coding",
                "description": "Programming and code-related queries"
            },
            {
                "name": "cooking_recipes",
                "description": "Cooking recipes with ingredients and instructions"
            },
            {
                "name": "translation",
                "description": "Translation requests between languages"
            },
            {
                "name": "creative_writing",
                "description": "Creative writing tasks"
            },
            {
                "name": "science_math",
                "description": "Scientific and mathematical calculations"
            },
            {
                "name": "url_lookup",
                "description": "URL content summarization"
            },
            {
                "name": "general",
                "description": "General queries not fitting other categories"
            }
        ]
    }

