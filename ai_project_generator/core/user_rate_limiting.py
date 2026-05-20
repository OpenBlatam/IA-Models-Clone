"""
User Rate Limiting - Rate Limiting por Usuario
==============================================

Rate limiting avanzado por usuario:
- Per-user rate limits
- Tier-based limits
- Dynamic limits
- Rate limit headers
- Quota management
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimitTier(str, Enum):
    """Tiers de rate limiting"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class UserRateLimiter:
    """
    Rate limiter por usuario.
    """
    
    def __init__(self) -> None:
        self.user_limits: Dict[str, Dict[str, Any]] = {}
        self.user_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "requests": 0,
            "window_start": datetime.now()
        })
        self.tier_limits: Dict[str, Dict[str, int]] = {
            RateLimitTier.FREE: {"requests": 100, "window": 3600},
            RateLimitTier.BASIC: {"requests": 1000, "window": 3600},
            RateLimitTier.PREMIUM: {"requests": 10000, "window": 3600},
            RateLimitTier.ENTERPRISE: {"requests": 100000, "window": 3600}
        }
    
    def set_user_tier(self, user_id: str, tier: str) -> None:
        """Establece tier de usuario"""
        if tier in self.tier_limits:
            self.user_limits[user_id] = {
                "tier": tier,
                **self.tier_limits[tier]
            }
            logger.info(f"User {user_id} set to tier {tier}")
    
    def set_custom_limit(
        self,
        user_id: str,
        requests: int,
        window: int
    ) -> None:
        """Establece límite personalizado"""
        self.user_limits[user_id] = {
            "tier": "custom",
            "requests": requests,
            "window": window
        }
        logger.info(f"Custom limit set for user {user_id}: {requests}/{window}s")
    
    def check_rate_limit(self, user_id: str) -> tuple[bool, Dict[str, Any]]:
        """Verifica rate limit"""
        if user_id not in self.user_limits:
            # Default: tier FREE
            limit = self.tier_limits[RateLimitTier.FREE]
        else:
            limit = self.user_limits[user_id]
        
        usage = self.user_usage[user_id]
        window_seconds = limit["window"]
        now = datetime.now()
        
        # Resetear ventana si ha expirado
        if (now - usage["window_start"]).total_seconds() > window_seconds:
            usage["requests"] = 0
            usage["window_start"] = now
        
        # Verificar límite
        allowed = usage["requests"] < limit["requests"]
        
        headers = {
            "X-RateLimit-Limit": str(limit["requests"]),
            "X-RateLimit-Remaining": str(max(0, limit["requests"] - usage["requests"])),
            "X-RateLimit-Reset": str(
                int((usage["window_start"] + timedelta(seconds=window_seconds)).timestamp())
            )
        }
        
        if allowed:
            usage["requests"] += 1
        
        return allowed, headers
    
    def get_user_quota(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene quota de usuario"""
        if user_id not in self.user_limits:
            return None
        
        limit = self.user_limits[user_id]
        usage = self.user_usage[user_id]
        
        return {
            "user_id": user_id,
            "tier": limit.get("tier", "free"),
            "limit": limit["requests"],
            "used": usage["requests"],
            "remaining": max(0, limit["requests"] - usage["requests"]),
            "window_seconds": limit["window"]
        }


def get_user_rate_limiter() -> UserRateLimiter:
    """Obtiene rate limiter por usuario"""
    return UserRateLimiter()

