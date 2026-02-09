"""
Advanced Rate Limiting endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.advanced_rate_limiting import AdvancedRateLimitingService, RateLimitStrategy

router = APIRouter()
rate_limit_service = AdvancedRateLimitingService()


@router.post("/create")
async def create_rate_limit(
    identifier: str,
    strategy: str,
    limit: int,
    window_seconds: int
) -> Dict[str, Any]:
    """Crear regla de rate limiting"""
    try:
        strategy_enum = RateLimitStrategy(strategy)
        rule = rate_limit_service.create_rate_limit(
            identifier, strategy_enum, limit, window_seconds
        )
        return {
            "identifier": rule.identifier,
            "strategy": rule.strategy.value,
            "limit": rule.limit,
            "window_seconds": rule.window_seconds,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check/{identifier}")
async def check_rate_limit(
    identifier: str,
    endpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Verificar rate limit"""
    try:
        result = rate_limit_service.check_rate_limit(identifier, endpoint)
        return {
            "allowed": result.allowed,
            "limit": result.limit,
            "remaining": result.remaining,
            "reset_at": result.reset_at.isoformat(),
            "retry_after": result.retry_after,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{identifier}")
async def get_rate_limit_status(identifier: str) -> Dict[str, Any]:
    """Obtener estado de rate limit"""
    try:
        status = rate_limit_service.get_rate_limit_status(identifier)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




