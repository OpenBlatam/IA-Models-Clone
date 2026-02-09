"""
Search API endpoints for finding conversations and maintenance records.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import re

from .base_router import BaseRouter
from ..utils.data_helpers import filter_by_fields, sort_by_function, paginate_items

# Create base router instance
base = BaseRouter(
    prefix="/api/search",
    tags=["Search"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, description="Search query")
    robot_type: Optional[str] = Field(None, description="Filter by robot type")
    maintenance_type: Optional[str] = Field(None, description="Filter by maintenance type")
    limit: int = Field(50, ge=1, le=500, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")


@router.post("/conversations")
@base.timed_endpoint("search_conversations")
async def search_conversations(
    request: SearchRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Search conversations by content.
    """
    base.log_request("search_conversations", query=request.query)
    
    all_conversations = base.database.get_all_conversations()
    
    # Apply filters using helper
    filtered = filter_by_fields(
        all_conversations,
        {
            "robot_type": request.robot_type,
            "maintenance_type": request.maintenance_type
        }
    )
    
    # Search in conversation messages
    query_lower = request.query.lower()
    results = []
    
    for conv in filtered:
        conv_id = conv.get("id")
        messages = base.database.get_messages_by_conversation(conv_id)
        
        # Search in message content
        matches = []
        for msg in messages:
            content = msg.get("content", "").lower()
            if query_lower in content:
                matches.append(msg)
        
        if matches:
            results.append({
                "conversation_id": conv_id,
                "robot_type": conv.get("robot_type"),
                "maintenance_type": conv.get("maintenance_type"),
                "created_at": conv.get("created_at"),
                "message_count": len(matches),
                "matched_messages": matches[:5]  # Limit matched messages
            })
    
    # Sort by relevance (number of matches) using helper
    results = sort_by_function(results, key_func=lambda x: x["message_count"], reverse=True)
    
    # Pagination using helper
    paginated_results, total, page = paginate_items(results, request.offset, request.limit)
    
    return base.paginated(
        items=paginated_results,
        total=total,
        page=page,
        page_size=request.limit,
        message=f"Found {total} conversations matching '{request.query}'"
    )


@router.get("/conversations")
async def search_conversations_get(
    q: str = Query(..., min_length=1, description="Search query"),
    robot_type: Optional[str] = Query(None, description="Filter by robot type"),
    maintenance_type: Optional[str] = Query(None, description="Filter by maintenance type"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Search conversations (GET version).
    """
    request = SearchRequest(
        query=q,
        robot_type=robot_type,
        maintenance_type=maintenance_type,
        limit=limit,
        offset=offset
    )
    return await search_conversations(request, _)


@router.get("/maintenance-records")
@base.timed_endpoint("search_maintenance_records")
async def search_maintenance_records(
    q: str = Query(..., min_length=1, description="Search query"),
    robot_type: Optional[str] = Query(None, description="Filter by robot type"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Search maintenance records.
    """
    base.log_request("search_maintenance_records", query=q, robot_type=robot_type)
    
    all_records = base.database.get_all_maintenance_records()
    
    # Apply filters using helper
    filtered = filter_by_fields(
        all_records,
        {"robot_type": robot_type}
    )
    
    # Search
    query_lower = q.lower()
    results = []
    
    for record in filtered:
        # Search in various fields
        description = record.get("description", "").lower()
        notes = record.get("notes", "").lower()
        robot_type_str = record.get("robot_type", "").lower()
        
        if (query_lower in description or 
            query_lower in notes or 
            query_lower in robot_type_str):
            results.append(record)
    
    # Pagination using helper
    paginated_results, total, page = paginate_items(results, offset, limit)
    
    return base.paginated(
        items=paginated_results,
        total=total,
        page=page,
        page_size=limit,
        message=f"Found {total} maintenance records matching '{q}'"
    )


@router.get("/suggestions")
@base.timed_endpoint("search_suggestions")
async def get_search_suggestions(
    q: str = Query(..., min_length=1, description="Partial query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum suggestions"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get search suggestions based on partial query.
    """
    base.log_request("search_suggestions", query=q)
    
    # Get common robot types and maintenance types
    conversations = base.database.get_all_conversations()
    
    robot_types = set()
    maintenance_types = set()
    keywords = set()
    
    for conv in conversations:
        if conv.get("robot_type"):
            robot_types.add(conv.get("robot_type"))
        if conv.get("maintenance_type"):
            maintenance_types.add(conv.get("maintenance_type"))
    
    # Get keywords from messages
    messages = base.database.get_all_messages()
    query_lower = q.lower()
    
    for msg in messages[:1000]:  # Limit for performance
        content = msg.get("content", "").lower()
        # Extract words that start with query
        words = re.findall(r'\b\w+\b', content)
        for word in words:
            if word.startswith(query_lower) and len(word) > len(query_lower):
                keywords.add(word)
    
    # Combine suggestions
    suggestions = []
    
    # Robot types
    for rt in robot_types:
        if q.lower() in rt.lower():
            suggestions.append({
                "type": "robot_type",
                "value": rt,
                "label": f"Robot: {rt}"
            })
    
    # Maintenance types
    for mt in maintenance_types:
        if q.lower() in mt.lower():
            suggestions.append({
                "type": "maintenance_type",
                "value": mt,
                "label": f"Maintenance: {mt}"
            })
    
    # Keywords
    for keyword in list(keywords)[:limit]:
        suggestions.append({
            "type": "keyword",
            "value": keyword,
            "label": keyword
        })
    
    return base.success({
        "query": q,
        "suggestions": suggestions[:limit]
    })




