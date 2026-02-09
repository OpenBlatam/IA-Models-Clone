"""
Authentication Service for Social Video Transcriber AI
Handles API key authentication and rate limiting
"""

import hashlib
import secrets
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class UserTier(str, Enum):
    """User subscription tiers"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class RateLimitConfig:
    """Rate limit configuration per tier"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    max_video_duration: int  # seconds
    max_batch_size: int
    priority: int  # Higher = better queue priority


TIER_LIMITS = {
    UserTier.FREE: RateLimitConfig(
        requests_per_minute=5,
        requests_per_hour=30,
        requests_per_day=100,
        max_video_duration=300,  # 5 minutes
        max_batch_size=3,
        priority=1,
    ),
    UserTier.BASIC: RateLimitConfig(
        requests_per_minute=15,
        requests_per_hour=100,
        requests_per_day=500,
        max_video_duration=1800,  # 30 minutes
        max_batch_size=10,
        priority=2,
    ),
    UserTier.PRO: RateLimitConfig(
        requests_per_minute=30,
        requests_per_hour=300,
        requests_per_day=2000,
        max_video_duration=3600,  # 1 hour
        max_batch_size=25,
        priority=3,
    ),
    UserTier.ENTERPRISE: RateLimitConfig(
        requests_per_minute=100,
        requests_per_hour=1000,
        requests_per_day=10000,
        max_video_duration=7200,  # 2 hours
        max_batch_size=100,
        priority=4,
    ),
}


@dataclass
class APIKey:
    """API Key definition"""
    key_id: str
    key_hash: str  # We store hash, not the actual key
    name: str
    user_id: str
    tier: UserTier
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Usage tracking
    total_requests: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "name": self.name,
            "user_id": self.user_id,
            "tier": self.tier.value,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "total_requests": self.total_requests,
        }


@dataclass 
class RateLimitState:
    """Rate limit tracking state"""
    minute_count: int = 0
    hour_count: int = 0
    day_count: int = 0
    minute_reset: datetime = field(default_factory=datetime.utcnow)
    hour_reset: datetime = field(default_factory=datetime.utcnow)
    day_reset: datetime = field(default_factory=datetime.utcnow)


class AuthService:
    """Service for API key authentication and rate limiting"""
    
    def __init__(self):
        self.settings = get_settings()
        self._api_keys: Dict[str, APIKey] = {}
        self._rate_limits: Dict[str, RateLimitState] = {}
        
        # Create a default development key
        if self.settings.environment == "development":
            self._create_dev_key()
    
    def _create_dev_key(self):
        """Create a development API key"""
        key_id = "dev_key"
        raw_key = "svt_dev_12345678901234567890"
        
        self._api_keys[key_id] = APIKey(
            key_id=key_id,
            key_hash=self._hash_key(raw_key),
            name="Development Key",
            user_id="dev_user",
            tier=UserTier.ENTERPRISE,
        )
        
        logger.info(f"Development API key created: {raw_key}")
    
    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for storage"""
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        tier: UserTier = UserTier.FREE,
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """
        Create a new API key
        
        Args:
            user_id: User identifier
            name: Key name/description
            tier: User tier
            expires_in_days: Optional expiration in days
            
        Returns:
            Tuple of (raw_key, APIKey object)
        """
        # Generate key
        key_id = secrets.token_hex(8)
        raw_key = f"svt_{secrets.token_hex(24)}"
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=self._hash_key(raw_key),
            name=name,
            user_id=user_id,
            tier=tier,
            expires_at=expires_at,
        )
        
        self._api_keys[key_id] = api_key
        logger.info(f"Created API key: {key_id} for user {user_id}")
        
        return raw_key, api_key
    
    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key
        
        Args:
            raw_key: Raw API key string
            
        Returns:
            APIKey if valid, None otherwise
        """
        if not raw_key:
            return None
        
        key_hash = self._hash_key(raw_key)
        
        for api_key in self._api_keys.values():
            if api_key.key_hash == key_hash:
                # Check if active
                if not api_key.active:
                    logger.warning(f"Inactive API key used: {api_key.key_id}")
                    return None
                
                # Check expiration
                if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                    logger.warning(f"Expired API key used: {api_key.key_id}")
                    return None
                
                # Update last used
                api_key.last_used_at = datetime.utcnow()
                api_key.total_requests += 1
                
                return api_key
        
        return None
    
    def check_rate_limit(self, api_key: APIKey) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits
        
        Args:
            api_key: API key object
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        limits = TIER_LIMITS[api_key.tier]
        now = datetime.utcnow()
        
        # Get or create rate limit state
        if api_key.key_id not in self._rate_limits:
            self._rate_limits[api_key.key_id] = RateLimitState()
        
        state = self._rate_limits[api_key.key_id]
        
        # Reset counters if needed
        if now >= state.minute_reset + timedelta(minutes=1):
            state.minute_count = 0
            state.minute_reset = now
        
        if now >= state.hour_reset + timedelta(hours=1):
            state.hour_count = 0
            state.hour_reset = now
        
        if now >= state.day_reset + timedelta(days=1):
            state.day_count = 0
            state.day_reset = now
        
        # Check limits
        rate_info = {
            "tier": api_key.tier.value,
            "minute": {
                "used": state.minute_count,
                "limit": limits.requests_per_minute,
                "reset_at": (state.minute_reset + timedelta(minutes=1)).isoformat(),
            },
            "hour": {
                "used": state.hour_count,
                "limit": limits.requests_per_hour,
                "reset_at": (state.hour_reset + timedelta(hours=1)).isoformat(),
            },
            "day": {
                "used": state.day_count,
                "limit": limits.requests_per_day,
                "reset_at": (state.day_reset + timedelta(days=1)).isoformat(),
            },
        }
        
        if state.minute_count >= limits.requests_per_minute:
            return False, {**rate_info, "exceeded": "minute"}
        
        if state.hour_count >= limits.requests_per_hour:
            return False, {**rate_info, "exceeded": "hour"}
        
        if state.day_count >= limits.requests_per_day:
            return False, {**rate_info, "exceeded": "day"}
        
        # Increment counters
        state.minute_count += 1
        state.hour_count += 1
        state.day_count += 1
        
        return True, rate_info
    
    def get_tier_limits(self, tier: UserTier) -> RateLimitConfig:
        """Get rate limits for a tier"""
        return TIER_LIMITS[tier]
    
    def revoke_api_key(self, key_id: str):
        """Revoke an API key"""
        if key_id in self._api_keys:
            self._api_keys[key_id].active = False
            logger.info(f"Revoked API key: {key_id}")
    
    def list_api_keys(self, user_id: Optional[str] = None) -> List[APIKey]:
        """List API keys, optionally filtered by user"""
        keys = list(self._api_keys.values())
        
        if user_id:
            keys = [k for k in keys if k.user_id == user_id]
        
        return keys
    
    def get_usage_stats(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for an API key"""
        if key_id not in self._api_keys:
            return None
        
        api_key = self._api_keys[key_id]
        state = self._rate_limits.get(key_id, RateLimitState())
        limits = TIER_LIMITS[api_key.tier]
        
        return {
            "key_id": key_id,
            "tier": api_key.tier.value,
            "total_requests": api_key.total_requests,
            "current_usage": {
                "minute": state.minute_count,
                "hour": state.hour_count,
                "day": state.day_count,
            },
            "limits": {
                "minute": limits.requests_per_minute,
                "hour": limits.requests_per_hour,
                "day": limits.requests_per_day,
            },
        }


_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get auth service singleton"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service












