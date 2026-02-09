"""
Retry API Endpoints
===================

Endpoints para retry manager y timeout manager.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from ..core.retry_manager import (
    get_retry_manager,
    RetryConfig,
    RetryStrategy
)
from ..core.timeout_manager import get_timeout_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/retry", tags=["retry"])


@router.get("/statistics")
async def get_retry_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de reintentos."""
    try:
        manager = get_retry_manager()
        stats = manager.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting retry statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeout/statistics")
async def get_timeout_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de timeouts."""
    try:
        manager = get_timeout_manager()
        stats = manager.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting timeout statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






