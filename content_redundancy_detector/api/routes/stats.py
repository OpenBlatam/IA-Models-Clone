"""
Stats Router - System statistics endpoints
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from services import get_system_stats
    from schemas import StatsResponse
except ImportError:
    logging.warning("services or schemas module not available")
    def get_system_stats(): return {}
    StatsResponse = Dict[str, Any]

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/stats", tags=["Stats"])


@router.get("", response_model=Dict[str, Any])
async def get_stats_endpoint() -> JSONResponse:
    """Get system statistics"""
    logger.info("System statistics requested")
    
    try:
        stats_data = get_system_stats()
        return JSONResponse(content={
            "success": True,
            "data": stats_data,
            "error": None
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")






