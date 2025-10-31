from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ..dependencies import (
from ..models import SEOResultModel
from ..operations import AsyncDatabaseOperations
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Database routes for Ultra-Optimized SEO Service v15.

This module contains database operation endpoints including:
- Store SEO results
- Retrieve SEO results
- Get analysis history
- Database statistics
"""


    get_async_db_operations,
    get_logger
)

# Create router with prefix and tags
router = APIRouter(
    prefix="/database",
    tags=["Database Operations"],
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"}
    }
)

@router.post("/store")
async def store_seo_result_endpoint(
    result: SEOResultModel,
    async_db_ops: AsyncDatabaseOperations = Depends(get_async_db_operations),
    logger = Depends(get_logger)
):
    """
    Store SEO analysis result using dedicated async database operations.
    
    Stores the SEO analysis result in the database with proper indexing
    and validation. Supports both Redis and MongoDB storage.
    """
    try:
        success = await async_db_ops.store_seo_result(result)
        if success:
            return {"success": True, "message": "SEO result stored successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store SEO result")
    except Exception as e:
        logger.error("Failed to store SEO result", error=str(e))
        raise HTTPException(status_code=500, detail="Database operation failed")

@router.get("/retrieve/{url:path}")
async def retrieve_seo_result_endpoint(
    url: str,
    async_db_ops: AsyncDatabaseOperations = Depends(get_async_db_operations),
    logger = Depends(get_logger)
):
    """
    Retrieve SEO analysis result using dedicated async database operations.
    
    Retrieves the most recent SEO analysis result for the given URL.
    Returns 404 if no result is found.
    """
    try:
        result = await async_db_ops.retrieve_seo_result(url)
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="SEO result not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve SEO result", error=str(e))
        raise HTTPException(status_code=500, detail="Database operation failed")

@router.get("/history/{url:path}")
async def get_analysis_history_endpoint(
    url: str,
    limit: int = 10,
    async_db_ops: AsyncDatabaseOperations = Depends(get_async_db_operations),
    logger = Depends(get_logger)
):
    """
    Get analysis history using dedicated async database operations.
    
    Retrieves historical SEO analysis results for the given URL.
    Results are sorted by timestamp in descending order.
    """
    try:
        history = await async_db_ops.get_analysis_history(url, limit)
        return {"url": url, "history": history, "count": len(history)}
    except Exception as e:
        logger.error("Failed to get analysis history", error=str(e))
        raise HTTPException(status_code=500, detail="Database operation failed")

@router.get("/stats")
async def get_database_stats_endpoint(
    async_db_ops: AsyncDatabaseOperations = Depends(get_async_db_operations),
    logger = Depends(get_logger)
):
    """
    Get database statistics using dedicated async database operations.
    
    Returns comprehensive database statistics including:
    - Total records count
    - Storage usage
    - Connection pool status
    - Performance metrics
    """
    try:
        stats = await async_db_ops.get_database_stats()
        return stats
    except Exception as e:
        logger.error("Failed to get database stats", error=str(e))
        raise HTTPException(status_code=500, detail="Database operation failed")

@router.delete("/cleanup")
async def cleanup_old_results_endpoint(
    days_old: int = 30,
    async_db_ops: AsyncDatabaseOperations = Depends(get_async_db_operations),
    logger = Depends(get_logger)
):
    """
    Clean up old analysis results.
    
    Deletes analysis results older than the specified number of days.
    Useful for maintaining database performance and storage efficiency.
    """
    try:
        deleted_count = await async_db_ops.delete_old_results(days_old)
        return {
            "success": True,
            "deleted_count": deleted_count,
            "days_old": days_old,
            "message": f"Deleted {deleted_count} old results"
        }
    except Exception as e:
        logger.error("Failed to cleanup old results", error=str(e))
        raise HTTPException(status_code=500, detail="Cleanup operation failed") 