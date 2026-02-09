"""
Rate Limit API Endpoints
========================

Endpoints para rate limiting avanzado y throttling.
"""

from fastapi import APIRouter, HTTPException, Query, Header
from typing import Dict, Any, Optional
import logging

from ..core.rate_limiter_advanced import (
    get_advanced_rate_limiter,
    RateLimitStrategy
)
from ..core.throttle_manager import get_throttle_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rate-limit", tags=["rate-limit"])


@router.post("/rules")
async def add_rate_limit_rule(
    rule_id: str,
    key: str,
    limit: int,
    window: float,
    strategy: str = "fixed_window"
) -> Dict[str, Any]:
    """Agregar regla de rate limiting."""
    try:
        limiter = get_advanced_rate_limiter()
        strategy_enum = RateLimitStrategy(strategy.lower())
        rule = limiter.add_rule(
            rule_id=rule_id,
            key=key,
            limit=limit,
            window=window,
            strategy=strategy_enum
        )
        return {
            "rule_id": rule.rule_id,
            "key": rule.key,
            "limit": rule.limit,
            "window": rule.window,
            "strategy": rule.strategy.value
        }
    except Exception as e:
        logger.error(f"Error adding rate limit rule: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules/{rule_id}/check")
async def check_rate_limit(
    rule_id: str,
    identifier: str = Header(..., alias="X-Identifier")
) -> Dict[str, Any]:
    """Verificar límite de rate limiting."""
    try:
        limiter = get_advanced_rate_limiter()
        result = limiter.check_limit(rule_id, identifier)
        return {
            "allowed": result.allowed,
            "limit": result.limit,
            "remaining": result.remaining,
            "reset_at": result.reset_at,
            "retry_after": result.retry_after
        }
    except Exception as e:
        logger.error(f"Error checking rate limit: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/throttle/rules")
async def add_throttle_rule(
    rule_id: str,
    key: str,
    max_requests: int,
    per_second: float,
    burst: int = 0
) -> Dict[str, Any]:
    """Agregar regla de throttling."""
    try:
        manager = get_throttle_manager()
        rule = manager.add_rule(
            rule_id=rule_id,
            key=key,
            max_requests=max_requests,
            per_second=per_second,
            burst=burst
        )
        return {
            "rule_id": rule.rule_id,
            "key": rule.key,
            "max_requests": rule.max_requests,
            "per_second": rule.per_second,
            "burst": rule.burst
        }
    except Exception as e:
        logger.error(f"Error adding throttle rule: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/throttle/{rule_id}/apply")
async def apply_throttle(
    rule_id: str,
    identifier: str = Header(..., alias="X-Identifier")
) -> Dict[str, Any]:
    """Aplicar throttling."""
    try:
        manager = get_throttle_manager()
        wait_time = await manager.throttle(rule_id, identifier)
        return {
            "rule_id": rule_id,
            "wait_time": wait_time,
            "throttled": wait_time > 0.0
        }
    except Exception as e:
        logger.error(f"Error applying throttle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






