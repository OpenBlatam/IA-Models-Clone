"""Search endpoints"""
from fastapi import APIRouter, Query
from typing import Optional
from utils.search_index import get_search_index

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("")
async def search_documents(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Search documents"""
    search_index = get_search_index()
    results = search_index.search(query, limit, offset)
    
    return {
        "query": query,
        "results": results,
        "total": len(results),
        "limit": limit,
        "offset": offset
    }

