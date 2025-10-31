"""
Advanced Search API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from ....services.advanced_search_service import AdvancedSearchService, SearchType, SearchScope, SearchFilter, SearchFacet
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str = Field(..., description="Search query")
    search_type: str = Field(default="text", description="Search type (text, semantic, fuzzy, boolean, advanced)")
    scope: str = Field(default="all", description="Search scope (title, content, tags, author, all)")
    filters: Optional[List[Dict[str, Any]]] = Field(default=None, description="Search filters")
    facets: Optional[List[Dict[str, Any]]] = Field(default=None, description="Search facets")
    sort: Optional[str] = Field(default=None, description="Sort field (prefix with - for descending)")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")


class SearchFilterRequest(BaseModel):
    """Request model for search filters."""
    field: str = Field(..., description="Field name")
    operator: str = Field(..., description="Operator (eq, ne, gt, lt, gte, lte, in, contains, starts_with, ends_with)")
    value: Any = Field(..., description="Filter value")
    boost: float = Field(default=1.0, description="Boost factor")


class SearchFacetRequest(BaseModel):
    """Request model for search facets."""
    field: str = Field(..., description="Field name")
    size: int = Field(default=10, ge=1, le=100, description="Number of facet values")
    min_count: int = Field(default=1, ge=1, description="Minimum count for facet values")


async def get_search_service(session: DatabaseSessionDep) -> AdvancedSearchService:
    """Get search service instance."""
    return AdvancedSearchService(session)


@router.post("/search", response_model=Dict[str, Any])
async def perform_search(
    request: SearchRequest = Depends(),
    search_service: AdvancedSearchService = Depends(get_search_service),
    current_user: CurrentUserDep = Depends()
):
    """Perform advanced search."""
    try:
        # Convert search type and scope to enums
        try:
            search_type = SearchType(request.search_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid search type: {request.search_type}")
        
        try:
            scope = SearchScope(request.scope.lower())
        except ValueError:
            raise ValidationError(f"Invalid search scope: {request.scope}")
        
        # Convert filters
        filters = None
        if request.filters:
            filters = [
                SearchFilter(
                    field=filter_dict["field"],
                    operator=filter_dict["operator"],
                    value=filter_dict["value"],
                    boost=filter_dict.get("boost", 1.0)
                )
                for filter_dict in request.filters
            ]
        
        # Convert facets
        facets = None
        if request.facets:
            facets = [
                SearchFacet(
                    field=facet_dict["field"],
                    size=facet_dict.get("size", 10),
                    min_count=facet_dict.get("min_count", 1)
                )
                for facet_dict in request.facets
            ]
        
        result = await search_service.search(
            query=request.query,
            search_type=search_type,
            scope=scope,
            filters=filters,
            facets=facets,
            sort=request.sort,
            page=request.page,
            page_size=request.page_size,
            user_id=str(current_user.id) if current_user else None
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Search completed successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform search"
        )


@router.get("/search", response_model=Dict[str, Any])
async def search_get(
    query: str = Query(..., description="Search query"),
    search_type: str = Query(default="text", description="Search type"),
    scope: str = Query(default="all", description="Search scope"),
    sort: Optional[str] = Query(default=None, description="Sort field"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
    search_service: AdvancedSearchService = Depends(get_search_service),
    current_user: CurrentUserDep = Depends()
):
    """Perform search via GET request."""
    try:
        # Convert search type and scope to enums
        try:
            search_type_enum = SearchType(search_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid search type: {search_type}")
        
        try:
            scope_enum = SearchScope(scope.lower())
        except ValueError:
            raise ValidationError(f"Invalid search scope: {scope}")
        
        result = await search_service.search(
            query=query,
            search_type=search_type_enum,
            scope=scope_enum,
            filters=None,
            facets=None,
            sort=sort,
            page=page,
            page_size=page_size,
            user_id=str(current_user.id) if current_user else None
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Search completed successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform search"
        )


@router.get("/suggestions", response_model=Dict[str, Any])
async def get_search_suggestions(
    query: str = Query(..., description="Search query for suggestions"),
    limit: int = Query(default=10, ge=1, le=50, description="Number of suggestions"),
    search_service: AdvancedSearchService = Depends(get_search_service),
    current_user: CurrentUserDep = Depends()
):
    """Get search suggestions based on query."""
    try:
        result = await search_service.get_search_suggestions(
            query=query,
            limit=limit
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Search suggestions retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get search suggestions"
        )


@router.get("/analytics", response_model=Dict[str, Any])
async def get_search_analytics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    search_service: AdvancedSearchService = Depends(get_search_service),
    current_user: CurrentUserDep = Depends()
):
    """Get search analytics."""
    try:
        result = await search_service.get_search_analytics(days=days)
        
        return {
            "success": True,
            "data": result,
            "message": "Search analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get search analytics"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_search_stats(
    search_service: AdvancedSearchService = Depends(get_search_service),
    current_user: CurrentUserDep = Depends()
):
    """Get search system statistics."""
    try:
        result = await search_service.get_search_stats()
        
        return {
            "success": True,
            "data": result,
            "message": "Search statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get search statistics"
        )


@router.post("/index/rebuild", response_model=Dict[str, Any])
async def rebuild_search_index(
    search_service: AdvancedSearchService = Depends(get_search_service),
    current_user: CurrentUserDep = Depends()
):
    """Rebuild search index."""
    try:
        result = await search_service.rebuild_search_index()
        
        return {
            "success": True,
            "data": result,
            "message": "Search index rebuilt successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rebuild search index"
        )


@router.get("/types", response_model=Dict[str, Any])
async def get_search_types():
    """Get available search types and their descriptions."""
    search_types = {
        "text": {
            "name": "Text Search",
            "description": "Traditional keyword-based search using SQL LIKE queries",
            "features": ["Keyword matching", "Case insensitive", "Partial matching"],
            "use_cases": ["Exact phrase search", "Simple keyword search"]
        },
        "semantic": {
            "name": "Semantic Search",
            "description": "AI-powered search using sentence embeddings and similarity",
            "features": ["Meaning-based search", "Context understanding", "Similarity scoring"],
            "use_cases": ["Finding similar content", "Concept-based search"]
        },
        "fuzzy": {
            "name": "Fuzzy Search",
            "description": "Search with typo tolerance and character variations",
            "features": ["Typo tolerance", "Character substitution", "Character insertion/deletion"],
            "use_cases": ["Handling typos", "Flexible matching"]
        },
        "boolean": {
            "name": "Boolean Search",
            "description": "Search using AND, OR, NOT operators for complex queries",
            "features": ["Boolean operators", "Complex query logic", "Precise control"],
            "use_cases": ["Complex search requirements", "Precise filtering"]
        },
        "advanced": {
            "name": "Advanced Search",
            "description": "Combines multiple search techniques for optimal results",
            "features": ["Hybrid approach", "Result merging", "Intelligent ranking"],
            "use_cases": ["Best overall results", "Comprehensive search"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "search_types": search_types,
            "total_types": len(search_types)
        },
        "message": "Search types retrieved successfully"
    }


@router.get("/scopes", response_model=Dict[str, Any])
async def get_search_scopes():
    """Get available search scopes and their descriptions."""
    search_scopes = {
        "title": {
            "name": "Title Only",
            "description": "Search only in post titles",
            "fields": ["title"]
        },
        "content": {
            "name": "Content Only",
            "description": "Search only in post content",
            "fields": ["content"]
        },
        "tags": {
            "name": "Tags Only",
            "description": "Search only in post tags",
            "fields": ["tags"]
        },
        "author": {
            "name": "Author Only",
            "description": "Search only in author names",
            "fields": ["author"]
        },
        "all": {
            "name": "All Fields",
            "description": "Search across all fields",
            "fields": ["title", "content", "tags", "author"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "search_scopes": search_scopes,
            "total_scopes": len(search_scopes)
        },
        "message": "Search scopes retrieved successfully"
    }


@router.get("/operators", response_model=Dict[str, Any])
async def get_search_operators():
    """Get available search operators for filters."""
    operators = {
        "eq": {
            "name": "Equals",
            "description": "Exact match",
            "example": "status = 'published'"
        },
        "ne": {
            "name": "Not Equals",
            "description": "Not equal to",
            "example": "status != 'draft'"
        },
        "gt": {
            "name": "Greater Than",
            "description": "Greater than",
            "example": "view_count > 100"
        },
        "lt": {
            "name": "Less Than",
            "description": "Less than",
            "example": "view_count < 1000"
        },
        "gte": {
            "name": "Greater Than or Equal",
            "description": "Greater than or equal to",
            "example": "view_count >= 100"
        },
        "lte": {
            "name": "Less Than or Equal",
            "description": "Less than or equal to",
            "example": "view_count <= 1000"
        },
        "in": {
            "name": "In",
            "description": "Value in list",
            "example": "category IN ['tech', 'science']"
        },
        "contains": {
            "name": "Contains",
            "description": "Contains substring",
            "example": "title CONTAINS 'python'"
        },
        "starts_with": {
            "name": "Starts With",
            "description": "Starts with substring",
            "example": "title STARTS WITH 'How to'"
        },
        "ends_with": {
            "name": "Ends With",
            "description": "Ends with substring",
            "example": "title ENDS WITH 'tutorial'"
        }
    }
    
    return {
        "success": True,
        "data": {
            "operators": operators,
            "total_operators": len(operators)
        },
        "message": "Search operators retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_search_health(
    search_service: AdvancedSearchService = Depends(get_search_service),
    current_user: CurrentUserDep = Depends()
):
    """Get search system health status."""
    try:
        # Get search stats
        stats = await search_service.get_search_stats()
        
        # Calculate health metrics
        total_searches = stats.get("total_searches", 0)
        today_searches = stats.get("today_searches", 0)
        indexed_documents = stats.get("indexed_documents", 0)
        cache_size = stats.get("cache_size", 0)
        
        # Check NLP models
        nlp_models = stats.get("nlp_models_loaded", {})
        models_loaded = sum(nlp_models.values())
        total_models = len(nlp_models)
        
        # Calculate health score
        health_score = 100
        
        # Check if models are loaded
        if models_loaded < total_models:
            health_score -= 20
        
        # Check if index is populated
        if indexed_documents == 0:
            health_score -= 30
        
        # Check search activity
        if today_searches == 0 and total_searches > 0:
            health_score -= 10
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_searches": total_searches,
                "today_searches": today_searches,
                "indexed_documents": indexed_documents,
                "cache_size": cache_size,
                "nlp_models_loaded": models_loaded,
                "total_nlp_models": total_models,
                "models_status": nlp_models,
                "timestamp": "2024-01-15T10:00:00Z"
            },
            "message": "Search health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get search health status"
        )


@router.get("/examples", response_model=Dict[str, Any])
async def get_search_examples():
    """Get search examples and usage patterns."""
    examples = {
        "text_search": {
            "description": "Simple keyword search",
            "examples": [
                "python tutorial",
                "machine learning",
                "web development"
            ]
        },
        "semantic_search": {
            "description": "Meaning-based search",
            "examples": [
                "how to learn programming",
                "best practices for web security",
                "data analysis techniques"
            ]
        },
        "fuzzy_search": {
            "description": "Typo-tolerant search",
            "examples": [
                "pythn tutorial",  # typo in python
                "machne learning",  # typo in machine
                "web devlopment"   # typo in development
            ]
        },
        "boolean_search": {
            "description": "Complex boolean queries",
            "examples": [
                "python AND tutorial",
                "machine learning OR AI",
                "web development NOT javascript"
            ]
        },
        "advanced_search": {
            "description": "Combined search techniques",
            "examples": [
                "python tutorial beginner",
                "machine learning best practices",
                "web development security"
            ]
        },
        "filtered_search": {
            "description": "Search with filters",
            "examples": [
                {
                    "query": "python",
                    "filters": [
                        {"field": "category", "operator": "eq", "value": "programming"},
                        {"field": "view_count", "operator": "gt", "value": 100}
                    ]
                }
            ]
        }
    }
    
    return {
        "success": True,
        "data": {
            "examples": examples,
            "total_examples": sum(len(v.get("examples", [])) for v in examples.values())
        },
        "message": "Search examples retrieved successfully"
    }
























