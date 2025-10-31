"""
Advanced Rate Limiting and Caching System for BUL API
Implements sophisticated rate limiting, caching, and performance optimization
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import time
import json
import hashlib
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import redis
import aioredis
from functools import wraps

logger = logging.getLogger(__name__)


class RateLimitType(str, Enum):
    """Rate limit types"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class CacheStrategy(str, Enum):
    """Cache strategies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class RateLimitRule(BaseModel):
    """Rate limit rule definition"""
    id: str = Field(..., description="Rule ID")
    name: str = Field(..., description="Rule name")
    endpoint: str = Field(..., description="Endpoint pattern")
    method: str = Field(..., description="HTTP method")
    limit_type: RateLimitType = Field(..., description="Rate limit type")
    requests_per_minute: int = Field(..., description="Requests per minute")
    requests_per_hour: int = Field(..., description="Requests per hour")
    requests_per_day: int = Field(..., description="Requests per day")
    burst_limit: int = Field(default=10, description="Burst limit")
    window_size: int = Field(default=60, description="Window size in seconds")
    is_active: bool = Field(default=True, description="Is rule active")
    user_tier: Optional[str] = Field(None, description="User tier restriction")
    ip_whitelist: List[str] = Field(default_factory=list, description="IP whitelist")
    ip_blacklist: List[str] = Field(default_factory=list, description="IP blacklist")


class CacheRule(BaseModel):
    """Cache rule definition"""
    id: str = Field(..., description="Rule ID")
    name: str = Field(..., description="Rule name")
    endpoint: str = Field(..., description="Endpoint pattern")
    method: str = Field(..., description="HTTP method")
    strategy: CacheStrategy = Field(..., description="Cache strategy")
    ttl: int = Field(default=300, description="Time to live in seconds")
    max_size: int = Field(default=1000, description="Maximum cache size")
    key_generator: Optional[str] = Field(None, description="Custom key generator")
    is_active: bool = Field(default=True, description="Is rule active")
    user_specific: bool = Field(default=False, description="Is cache user-specific")
    vary_headers: List[str] = Field(default_factory=list, description="Headers to vary cache by")


@dataclass
class RateLimitInfo:
    """Rate limit information"""
    limit: int
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None


@dataclass
class CacheEntry:
    """Cache entry"""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size: int = 0


class TokenBucket:
    """Token bucket rate limiter"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket"""
        now = time.time()
        time_passed = now - self.last_refill
        
        # Refill tokens
        self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
        self.last_refill = now
        
        # Check if enough tokens available
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def get_retry_after(self) -> float:
        """Get retry after time"""
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.refill_rate


class SlidingWindow:
    """Sliding window rate limiter"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
    
    def is_allowed(self) -> Tuple[bool, int]:
        """Check if request is allowed"""
        now = time.time()
        
        # Remove old requests
        while self.requests and self.requests[0] <= now - self.window_size:
            self.requests.popleft()
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True, 0
        
        # Calculate retry after
        oldest_request = self.requests[0]
        retry_after = int(oldest_request + self.window_size - now)
        return False, retry_after


class AdvancedRateLimiter:
    """Advanced rate limiting system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.rules: Dict[str, RateLimitRule] = {}
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.sliding_windows: Dict[str, SlidingWindow] = {}
        self._initialize_default_rules()
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Rate limiter Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory rate limiting: {e}")
            self.redis_client = None
    
    def _initialize_default_rules(self):
        """Initialize default rate limiting rules"""
        default_rules = [
            RateLimitRule(
                id="default_api",
                name="Default API Rate Limit",
                endpoint="*",
                method="*",
                limit_type=RateLimitType.SLIDING_WINDOW,
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_limit=10
            ),
            RateLimitRule(
                id="document_generation",
                name="Document Generation Rate Limit",
                endpoint="/generate/*",
                method="POST",
                limit_type=RateLimitType.TOKEN_BUCKET,
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=500,
                burst_limit=3
            ),
            RateLimitRule(
                id="ai_requests",
                name="AI Model Requests Rate Limit",
                endpoint="/ai/*",
                method="*",
                limit_type=RateLimitType.SLIDING_WINDOW,
                requests_per_minute=30,
                requests_per_hour=500,
                requests_per_day=2000,
                burst_limit=5
            ),
            RateLimitRule(
                id="analytics",
                name="Analytics Rate Limit",
                endpoint="/analytics/*",
                method="GET",
                limit_type=RateLimitType.FIXED_WINDOW,
                requests_per_minute=20,
                requests_per_hour=200,
                requests_per_day=1000,
                burst_limit=5
            ),
            RateLimitRule(
                id="premium_users",
                name="Premium Users Rate Limit",
                endpoint="*",
                method="*",
                limit_type=RateLimitType.SLIDING_WINDOW,
                requests_per_minute=120,
                requests_per_hour=2000,
                requests_per_day=20000,
                burst_limit=20,
                user_tier="premium"
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
        
        logger.info(f"Initialized {len(default_rules)} rate limiting rules")
    
    async def check_rate_limit(
        self,
        endpoint: str,
        method: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_tier: Optional[str] = None
    ) -> RateLimitInfo:
        """Check rate limit for request"""
        # Find applicable rules
        applicable_rules = self._find_applicable_rules(endpoint, method, user_tier)
        
        if not applicable_rules:
            # No rate limiting
            return RateLimitInfo(limit=0, remaining=0, reset_time=0)
        
        # Use the most restrictive rule
        rule = min(applicable_rules, key=lambda r: r.requests_per_minute)
        
        # Check IP restrictions
        if ip_address:
            if ip_address in rule.ip_blacklist:
                return RateLimitInfo(limit=0, remaining=0, reset_time=0, retry_after=3600)
            if rule.ip_whitelist and ip_address not in rule.ip_whitelist:
                return RateLimitInfo(limit=0, remaining=0, reset_time=0, retry_after=3600)
        
        # Generate key for rate limiting
        key = self._generate_rate_limit_key(rule, user_id, ip_address)
        
        # Check rate limit based on type
        if rule.limit_type == RateLimitType.TOKEN_BUCKET:
            return await self._check_token_bucket(rule, key)
        elif rule.limit_type == RateLimitType.SLIDING_WINDOW:
            return await self._check_sliding_window(rule, key)
        elif rule.limit_type == RateLimitType.FIXED_WINDOW:
            return await self._check_fixed_window(rule, key)
        else:
            return RateLimitInfo(limit=0, remaining=0, reset_time=0)
    
    def _find_applicable_rules(
        self,
        endpoint: str,
        method: str,
        user_tier: Optional[str] = None
    ) -> List[RateLimitRule]:
        """Find applicable rate limiting rules"""
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.is_active:
                continue
            
            # Check endpoint pattern
            if not self._matches_pattern(rule.endpoint, endpoint):
                continue
            
            # Check method
            if rule.method != "*" and rule.method.upper() != method.upper():
                continue
            
            # Check user tier
            if rule.user_tier and rule.user_tier != user_tier:
                continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    def _matches_pattern(self, pattern: str, endpoint: str) -> bool:
        """Check if endpoint matches pattern"""
        if pattern == "*":
            return True
        
        # Simple wildcard matching
        if "*" in pattern:
            import fnmatch
            return fnmatch.fnmatch(endpoint, pattern)
        
        return pattern == endpoint
    
    def _generate_rate_limit_key(
        self,
        rule: RateLimitRule,
        user_id: Optional[str],
        ip_address: Optional[str]
    ) -> str:
        """Generate rate limit key"""
        if user_id:
            return f"rate_limit:{rule.id}:user:{user_id}"
        elif ip_address:
            return f"rate_limit:{rule.id}:ip:{ip_address}"
        else:
            return f"rate_limit:{rule.id}:global"
    
    async def _check_token_bucket(self, rule: RateLimitRule, key: str) -> RateLimitInfo:
        """Check token bucket rate limit"""
        if self.redis_client:
            return await self._check_token_bucket_redis(rule, key)
        else:
            return await self._check_token_bucket_memory(rule, key)
    
    async def _check_token_bucket_redis(self, rule: RateLimitRule, key: str) -> RateLimitInfo:
        """Check token bucket using Redis"""
        try:
            # Get current bucket state
            bucket_data = await self.redis_client.hgetall(key)
            
            if not bucket_data:
                # Initialize bucket
                tokens = rule.burst_limit
                last_refill = time.time()
            else:
                tokens = float(bucket_data.get(b"tokens", rule.burst_limit))
                last_refill = float(bucket_data.get(b"last_refill", time.time()))
            
            # Refill tokens
            now = time.time()
            time_passed = now - last_refill
            refill_rate = rule.requests_per_minute / 60.0
            tokens = min(rule.burst_limit, tokens + time_passed * refill_rate)
            
            # Check if request allowed
            if tokens >= 1:
                tokens -= 1
                allowed = True
                retry_after = None
            else:
                allowed = False
                retry_after = int((1 - tokens) / refill_rate)
            
            # Update bucket state
            await self.redis_client.hset(key, mapping={
                "tokens": tokens,
                "last_refill": now
            })
            await self.redis_client.expire(key, 3600)  # 1 hour expiry
            
            return RateLimitInfo(
                limit=rule.burst_limit,
                remaining=int(tokens),
                reset_time=int(now + 60),  # Reset in 1 minute
                retry_after=retry_after
            )
            
        except Exception as e:
            logger.error(f"Redis token bucket error: {e}")
            return RateLimitInfo(limit=0, remaining=0, reset_time=0)
    
    async def _check_token_bucket_memory(self, rule: RateLimitRule, key: str) -> RateLimitInfo:
        """Check token bucket using memory"""
        if key not in self.token_buckets:
            self.token_buckets[key] = TokenBucket(
                capacity=rule.burst_limit,
                refill_rate=rule.requests_per_minute / 60.0
            )
        
        bucket = self.token_buckets[key]
        allowed = bucket.consume(1)
        
        if allowed:
            retry_after = None
        else:
            retry_after = int(bucket.get_retry_after())
        
        return RateLimitInfo(
            limit=rule.burst_limit,
            remaining=int(bucket.tokens),
            reset_time=int(time.time() + 60),
            retry_after=retry_after
        )
    
    async def _check_sliding_window(self, rule: RateLimitRule, key: str) -> RateLimitInfo:
        """Check sliding window rate limit"""
        if self.redis_client:
            return await self._check_sliding_window_redis(rule, key)
        else:
            return await self._check_sliding_window_memory(rule, key)
    
    async def _check_sliding_window_redis(self, rule: RateLimitRule, key: str) -> RateLimitInfo:
        """Check sliding window using Redis"""
        try:
            now = time.time()
            window_start = now - rule.window_size
            
            # Remove old requests
            await self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            current_requests = await self.redis_client.zcard(key)
            
            if current_requests < rule.requests_per_minute:
                # Allow request
                await self.redis_client.zadd(key, {str(now): now})
                await self.redis_client.expire(key, rule.window_size)
                
                return RateLimitInfo(
                    limit=rule.requests_per_minute,
                    remaining=rule.requests_per_minute - current_requests - 1,
                    reset_time=int(now + rule.window_size)
                )
            else:
                # Rate limited
                oldest_request = await self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest_request:
                    retry_after = int(oldest_request[0][1] + rule.window_size - now)
                else:
                    retry_after = rule.window_size
                
                return RateLimitInfo(
                    limit=rule.requests_per_minute,
                    remaining=0,
                    reset_time=int(now + retry_after),
                    retry_after=retry_after
                )
                
        except Exception as e:
            logger.error(f"Redis sliding window error: {e}")
            return RateLimitInfo(limit=0, remaining=0, reset_time=0)
    
    async def _check_sliding_window_memory(self, rule: RateLimitRule, key: str) -> RateLimitInfo:
        """Check sliding window using memory"""
        if key not in self.sliding_windows:
            self.sliding_windows[key] = SlidingWindow(
                window_size=rule.window_size,
                max_requests=rule.requests_per_minute
            )
        
        window = self.sliding_windows[key]
        allowed, retry_after = window.is_allowed()
        
        return RateLimitInfo(
            limit=rule.requests_per_minute,
            remaining=rule.requests_per_minute - len(window.requests),
            reset_time=int(time.time() + rule.window_size),
            retry_after=retry_after if not allowed else None
        )
    
    async def _check_fixed_window(self, rule: RateLimitRule, key: str) -> RateLimitInfo:
        """Check fixed window rate limit"""
        if self.redis_client:
            return await self._check_fixed_window_redis(rule, key)
        else:
            return await self._check_fixed_window_memory(rule, key)
    
    async def _check_fixed_window_redis(self, rule: RateLimitRule, key: str) -> RateLimitInfo:
        """Check fixed window using Redis"""
        try:
            now = time.time()
            window_start = int(now // rule.window_size) * rule.window_size
            window_key = f"{key}:{window_start}"
            
            # Get current count
            current_count = await self.redis_client.get(window_key)
            current_count = int(current_count) if current_count else 0
            
            if current_count < rule.requests_per_minute:
                # Allow request
                await self.redis_client.incr(window_key)
                await self.redis_client.expire(window_key, rule.window_size)
                
                return RateLimitInfo(
                    limit=rule.requests_per_minute,
                    remaining=rule.requests_per_minute - current_count - 1,
                    reset_time=int(window_start + rule.window_size)
                )
            else:
                # Rate limited
                return RateLimitInfo(
                    limit=rule.requests_per_minute,
                    remaining=0,
                    reset_time=int(window_start + rule.window_size),
                    retry_after=int(window_start + rule.window_size - now)
                )
                
        except Exception as e:
            logger.error(f"Redis fixed window error: {e}")
            return RateLimitInfo(limit=0, remaining=0, reset_time=0)
    
    async def _check_fixed_window_memory(self, rule: RateLimitRule, key: str) -> RateLimitInfo:
        """Check fixed window using memory"""
        now = time.time()
        window_start = int(now // rule.window_size) * rule.window_size
        window_key = f"{key}:{window_start}"
        
        if window_key not in self.sliding_windows:
            self.sliding_windows[window_key] = SlidingWindow(
                window_size=rule.window_size,
                max_requests=rule.requests_per_minute
            )
        
        window = self.sliding_windows[window_key]
        allowed, retry_after = window.is_allowed()
        
        return RateLimitInfo(
            limit=rule.requests_per_minute,
            remaining=rule.requests_per_minute - len(window.requests),
            reset_time=int(window_start + rule.window_size),
            retry_after=retry_after if not allowed else None
        )
    
    async def add_rule(self, rule: RateLimitRule):
        """Add rate limiting rule"""
        self.rules[rule.id] = rule
        logger.info(f"Added rate limiting rule: {rule.id}")
    
    async def remove_rule(self, rule_id: str):
        """Remove rate limiting rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed rate limiting rule: {rule_id}")
    
    async def get_rule(self, rule_id: str) -> Optional[RateLimitRule]:
        """Get rate limiting rule"""
        return self.rules.get(rule_id)
    
    async def list_rules(self) -> List[RateLimitRule]:
        """List all rate limiting rules"""
        return list(self.rules.values())


class AdvancedCache:
    """Advanced caching system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", max_memory_size: int = 1000):
        self.redis_url = redis_url
        self.redis_client = None
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.max_memory_size = max_memory_size
        self.rules: Dict[str, CacheRule] = {}
        self._initialize_default_rules()
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Cache Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory caching: {e}")
            self.redis_client = None
    
    def _initialize_default_rules(self):
        """Initialize default caching rules"""
        default_rules = [
            CacheRule(
                id="default_api",
                name="Default API Cache",
                endpoint="*",
                method="GET",
                strategy=CacheStrategy.TTL,
                ttl=300,
                max_size=1000
            ),
            CacheRule(
                id="document_templates",
                name="Document Templates Cache",
                endpoint="/templates/*",
                method="GET",
                strategy=CacheStrategy.TTL,
                ttl=3600,
                max_size=100,
                user_specific=False
            ),
            CacheRule(
                id="user_data",
                name="User Data Cache",
                endpoint="/user/*",
                method="GET",
                strategy=CacheStrategy.TTL,
                ttl=600,
                max_size=500,
                user_specific=True
            ),
            CacheRule(
                id="analytics",
                name="Analytics Cache",
                endpoint="/analytics/*",
                method="GET",
                strategy=CacheStrategy.TTL,
                ttl=1800,
                max_size=200
            ),
            CacheRule(
                id="model_analytics",
                name="Model Analytics Cache",
                endpoint="/models/*/analytics",
                method="GET",
                strategy=CacheStrategy.TTL,
                ttl=900,
                max_size=100
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
        
        logger.info(f"Initialized {len(default_rules)} caching rules")
    
    async def get(
        self,
        key: str,
        endpoint: str,
        method: str,
        user_id: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from cache"""
        # Find applicable cache rule
        rule = self._find_applicable_rule(endpoint, method)
        if not rule or not rule.is_active:
            return None
        
        # Generate cache key
        cache_key = self._generate_cache_key(rule, key, user_id)
        
        if self.redis_client:
            return await self._get_from_redis(cache_key, rule)
        else:
            return await self._get_from_memory(cache_key, rule)
    
    async def set(
        self,
        key: str,
        value: Any,
        endpoint: str,
        method: str,
        user_id: Optional[str] = None
    ):
        """Set value in cache"""
        # Find applicable cache rule
        rule = self._find_applicable_rule(endpoint, method)
        if not rule or not rule.is_active:
            return
        
        # Generate cache key
        cache_key = self._generate_cache_key(rule, key, user_id)
        
        if self.redis_client:
            await self._set_in_redis(cache_key, value, rule)
        else:
            await self._set_in_memory(cache_key, value, rule)
    
    def _find_applicable_rule(self, endpoint: str, method: str) -> Optional[CacheRule]:
        """Find applicable cache rule"""
        for rule in self.rules.values():
            if not rule.is_active:
                continue
            
            # Check endpoint pattern
            if not self._matches_pattern(rule.endpoint, endpoint):
                continue
            
            # Check method
            if rule.method != "*" and rule.method.upper() != method.upper():
                continue
            
            return rule
        
        return None
    
    def _matches_pattern(self, pattern: str, endpoint: str) -> bool:
        """Check if endpoint matches pattern"""
        if pattern == "*":
            return True
        
        # Simple wildcard matching
        if "*" in pattern:
            import fnmatch
            return fnmatch.fnmatch(endpoint, pattern)
        
        return pattern == endpoint
    
    def _generate_cache_key(self, rule: CacheRule, key: str, user_id: Optional[str]) -> str:
        """Generate cache key"""
        if rule.user_specific and user_id:
            return f"cache:{rule.id}:user:{user_id}:{key}"
        else:
            return f"cache:{rule.id}:{key}"
    
    async def _get_from_redis(self, cache_key: str, rule: CacheRule) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            value = await self.redis_client.get(cache_key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            return None
    
    async def _set_in_redis(self, cache_key: str, value: Any, rule: CacheRule):
        """Set value in Redis cache"""
        try:
            serialized_value = json.dumps(value)
            await self.redis_client.setex(cache_key, rule.ttl, serialized_value)
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
    
    async def _get_from_memory(self, cache_key: str, rule: CacheRule) -> Optional[Any]:
        """Get value from memory cache"""
        if cache_key not in self.memory_cache:
            return None
        
        entry = self.memory_cache[cache_key]
        
        # Check if expired
        if datetime.utcnow() > entry.expires_at:
            del self.memory_cache[cache_key]
            return None
        
        # Update access info
        entry.access_count += 1
        entry.last_accessed = datetime.utcnow()
        
        return entry.value
    
    async def _set_in_memory(self, cache_key: str, value: Any, rule: CacheRule):
        """Set value in memory cache"""
        # Check cache size limit
        if len(self.memory_cache) >= self.max_memory_size:
            await self._evict_entries(rule)
        
        # Create cache entry
        now = datetime.utcnow()
        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=now,
            expires_at=now + timedelta(seconds=rule.ttl),
            size=len(str(value))
        )
        
        self.memory_cache[cache_key] = entry
    
    async def _evict_entries(self, rule: CacheRule):
        """Evict entries based on strategy"""
        if rule.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].last_accessed
            )
            del self.memory_cache[oldest_key]
        
        elif rule.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            least_used_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].access_count
            )
            del self.memory_cache[least_used_key]
        
        else:
            # Remove oldest
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].created_at
            )
            del self.memory_cache[oldest_key]
    
    async def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        if self.redis_client:
            await self._invalidate_redis(pattern)
        else:
            await self._invalidate_memory(pattern)
    
    async def _invalidate_redis(self, pattern: str):
        """Invalidate Redis cache entries"""
        try:
            keys = await self.redis_client.keys(f"cache:*{pattern}*")
            if keys:
                await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis cache invalidation error: {e}")
    
    async def _invalidate_memory(self, pattern: str):
        """Invalidate memory cache entries"""
        keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.memory_cache[key]
    
    async def clear(self):
        """Clear all cache"""
        if self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis cache clear error: {e}")
        else:
            self.memory_cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.redis_client:
            try:
                info = await self.redis_client.info("memory")
                return {
                    "type": "redis",
                    "memory_used": info.get("used_memory_human", "0B"),
                    "keys": await self.redis_client.dbsize(),
                    "hit_rate": "N/A"  # Would need to track this
                }
            except Exception as e:
                logger.error(f"Redis cache stats error: {e}")
                return {"type": "redis", "error": str(e)}
        else:
            total_size = sum(entry.size for entry in self.memory_cache.values())
            return {
                "type": "memory",
                "entries": len(self.memory_cache),
                "total_size": total_size,
                "max_size": self.max_memory_size
            }


# Global instances
rate_limiter = AdvancedRateLimiter()
cache = AdvancedCache()


# Decorators for easy use
def rate_limit(endpoint: str = "*", method: str = "*"):
    """Rate limiting decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request info (this would be adapted based on your framework)
            request = kwargs.get('request')
            if not request:
                return await func(*args, **kwargs)
            
            user_id = getattr(request, 'user_id', None)
            ip_address = getattr(request, 'client_ip', None)
            user_tier = getattr(request, 'user_tier', None)
            
            # Check rate limit
            rate_limit_info = await rate_limiter.check_rate_limit(
                endpoint=endpoint,
                method=method,
                user_id=user_id,
                ip_address=ip_address,
                user_tier=user_tier
            )
            
            if rate_limit_info.retry_after:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(rate_limit_info.limit),
                        "X-RateLimit-Remaining": str(rate_limit_info.remaining),
                        "X-RateLimit-Reset": str(rate_limit_info.reset_time),
                        "Retry-After": str(rate_limit_info.retry_after)
                    }
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def cache_response(endpoint: str = "*", method: str = "GET", ttl: int = 300):
    """Cache response decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request info
            request = kwargs.get('request')
            if not request:
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = f"{endpoint}:{method}:{hash(str(request.url))}"
            user_id = getattr(request, 'user_id', None)
            
            # Try to get from cache
            cached_result = await cache.get(
                key=cache_key,
                endpoint=endpoint,
                method=method,
                user_id=user_id
            )
            
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(
                key=cache_key,
                value=result,
                endpoint=endpoint,
                method=method,
                user_id=user_id
            )
            
            return result
        
        return wrapper
    return decorator














