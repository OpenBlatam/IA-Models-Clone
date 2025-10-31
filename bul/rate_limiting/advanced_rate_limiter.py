"""
BUL Advanced Rate Limiting System
=================================

Advanced rate limiting with Redis backend, multiple algorithms, and dynamic limits.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import Request, HTTPException
from pydantic import BaseModel, Field
import redis.asyncio as redis
import hashlib

from ..utils import get_logger, get_cache_manager
from ..config import get_config

logger = get_logger(__name__)

class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"

class RateLimitScope(str, Enum):
    """Rate limiting scopes"""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    API_KEY = "api_key"
    CUSTOM = "custom"

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    scope: RateLimitScope
    algorithm: RateLimitAlgorithm
    requests_per_window: int
    window_size_seconds: int
    burst_limit: Optional[int] = None
    cost_per_request: int = 1
    enabled: bool = True
    priority: int = 0
    conditions: Dict[str, Any] = None
    actions: List[str] = None

@dataclass
class RateLimitResult:
    """Rate limiting result"""
    allowed: bool
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    algorithm: RateLimitAlgorithm = None
    scope: RateLimitScope = None
    identifier: str = None
    cost: int = 1

class AdvancedRateLimiter:
    """Advanced rate limiting system with Redis backend"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.cache_manager = get_cache_manager()
        
        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        self.redis_connected = False
        
        # Rate limiting rules
        self.rules: Dict[str, RateLimitRule] = {}
        self.default_rules: List[RateLimitRule] = []
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "blocked_requests": 0,
            "rule_hits": {},
            "algorithm_usage": {}
        }
        
        # Initialize default rules
        self._initialize_default_rules()
    
    async def initialize(self):
        """Initialize rate limiter with Redis connection"""
        try:
            # Connect to Redis if configured
            if self.config.cache.backend == "redis":
                self.redis_client = redis.from_url(
                    self.config.cache.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                
                # Test connection
                await self.redis_client.ping()
                self.redis_connected = True
                self.logger.info("Rate limiter connected to Redis")
            else:
                self.logger.info("Rate limiter using in-memory storage")
            
            # Load rules from configuration
            await self._load_rules()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize rate limiter: {e}")
            self.redis_connected = False
            return False
    
    async def close(self):
        """Close rate limiter connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            self.logger.info("Rate limiter closed")
        except Exception as e:
            self.logger.error(f"Error closing rate limiter: {e}")
    
    def _initialize_default_rules(self):
        """Initialize default rate limiting rules"""
        self.default_rules = [
            RateLimitRule(
                name="global_rate_limit",
                scope=RateLimitScope.GLOBAL,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                requests_per_window=1000,
                window_size_seconds=3600,  # 1 hour
                priority=1
            ),
            RateLimitRule(
                name="api_rate_limit",
                scope=RateLimitScope.ENDPOINT,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                requests_per_window=100,
                window_size_seconds=60,  # 1 minute
                burst_limit=20,
                priority=2
            ),
            RateLimitRule(
                name="user_rate_limit",
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                requests_per_window=500,
                window_size_seconds=3600,  # 1 hour
                priority=3
            ),
            RateLimitRule(
                name="ip_rate_limit",
                scope=RateLimitScope.IP,
                algorithm=RateLimitAlgorithm.ADAPTIVE,
                requests_per_window=200,
                window_size_seconds=3600,  # 1 hour
                priority=4
            ),
            RateLimitRule(
                name="document_generation_limit",
                scope=RateLimitScope.ENDPOINT,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                requests_per_window=10,
                window_size_seconds=60,  # 1 minute
                cost_per_request=5,
                conditions={"endpoint": "/generate"},
                priority=5
            )
        ]
    
    async def _load_rules(self):
        """Load rate limiting rules from configuration"""
        try:
            # Load default rules
            for rule in self.default_rules:
                self.rules[rule.name] = rule
            
            # Load custom rules from configuration (if any)
            custom_rules = getattr(self.config, 'rate_limiting', {}).get('rules', [])
            for rule_config in custom_rules:
                rule = RateLimitRule(**rule_config)
                self.rules[rule.name] = rule
            
            self.logger.info(f"Loaded {len(self.rules)} rate limiting rules")
        
        except Exception as e:
            self.logger.error(f"Error loading rate limiting rules: {e}")
    
    async def check_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope = RateLimitScope.IP,
        endpoint: str = None,
        cost: int = 1,
        custom_rules: List[str] = None
    ) -> RateLimitResult:
        """Check rate limit for identifier"""
        try:
            self.stats["total_requests"] += 1
            
            # Get applicable rules
            applicable_rules = self._get_applicable_rules(scope, endpoint, custom_rules)
            
            if not applicable_rules:
                # No rules apply, allow request
                return RateLimitResult(
                    allowed=True,
                    limit=0,
                    remaining=0,
                    reset_time=datetime.now() + timedelta(seconds=60),
                    identifier=identifier,
                    cost=cost
                )
            
            # Check each rule (in priority order)
            for rule in sorted(applicable_rules, key=lambda x: x.priority):
                result = await self._check_rule(rule, identifier, cost)
                
                if not result.allowed:
                    self.stats["blocked_requests"] += 1
                    self.stats["rule_hits"][rule.name] = self.stats["rule_hits"].get(rule.name, 0) + 1
                    return result
            
            # All rules passed
            self.stats["allowed_requests"] += 1
            return RateLimitResult(
                allowed=True,
                limit=applicable_rules[0].requests_per_window,
                remaining=applicable_rules[0].requests_per_window - cost,
                reset_time=datetime.now() + timedelta(seconds=applicable_rules[0].window_size_seconds),
                algorithm=applicable_rules[0].algorithm,
                scope=scope,
                identifier=identifier,
                cost=cost
            )
        
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {e}")
            # Fail open - allow request if rate limiter fails
            return RateLimitResult(
                allowed=True,
                limit=0,
                remaining=0,
                reset_time=datetime.now() + timedelta(seconds=60),
                identifier=identifier,
                cost=cost
            )
    
    def _get_applicable_rules(
        self,
        scope: RateLimitScope,
        endpoint: str = None,
        custom_rules: List[str] = None
    ) -> List[RateLimitRule]:
        """Get applicable rate limiting rules"""
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check scope match
            if rule.scope != scope and rule.scope != RateLimitScope.GLOBAL:
                continue
            
            # Check custom rules
            if custom_rules and rule.name not in custom_rules:
                continue
            
            # Check endpoint conditions
            if rule.conditions and endpoint:
                if "endpoint" in rule.conditions:
                    if not endpoint.startswith(rule.conditions["endpoint"]):
                        continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    async def _check_rule(self, rule: RateLimitRule, identifier: str, cost: int) -> RateLimitResult:
        """Check a specific rate limiting rule"""
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(rule, identifier)
            
            # Apply algorithm-specific logic
            if rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                return await self._check_fixed_window(rule, cache_key, cost)
            elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return await self._check_sliding_window(rule, cache_key, cost)
            elif rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                return await self._check_token_bucket(rule, cache_key, cost)
            elif rule.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                return await self._check_leaky_bucket(rule, cache_key, cost)
            elif rule.algorithm == RateLimitAlgorithm.ADAPTIVE:
                return await self._check_adaptive(rule, cache_key, cost)
            else:
                # Default to fixed window
                return await self._check_fixed_window(rule, cache_key, cost)
        
        except Exception as e:
            self.logger.error(f"Error checking rule {rule.name}: {e}")
            # Fail open
            return RateLimitResult(
                allowed=True,
                limit=rule.requests_per_window,
                remaining=rule.requests_per_window - cost,
                reset_time=datetime.now() + timedelta(seconds=rule.window_size_seconds),
                algorithm=rule.algorithm,
                scope=rule.scope,
                identifier=identifier,
                cost=cost
            )
    
    async def _check_fixed_window(self, rule: RateLimitRule, cache_key: str, cost: int) -> RateLimitResult:
        """Fixed window rate limiting algorithm"""
        try:
            current_time = int(time.time())
            window_start = current_time - (current_time % rule.window_size_seconds)
            window_key = f"{cache_key}:{window_start}"
            
            if self.redis_connected:
                # Use Redis
                current_count = await self.redis_client.get(window_key)
                current_count = int(current_count) if current_count else 0
                
                if current_count + cost > rule.requests_per_window:
                    return RateLimitResult(
                        allowed=False,
                        limit=rule.requests_per_window,
                        remaining=0,
                        reset_time=datetime.fromtimestamp(window_start + rule.window_size_seconds),
                        retry_after=rule.window_size_seconds - (current_time - window_start),
                        algorithm=rule.algorithm,
                        scope=rule.scope,
                        identifier=cache_key,
                        cost=cost
                    )
                
                # Increment counter
                await self.redis_client.incrby(window_key, cost)
                await self.redis_client.expire(window_key, rule.window_size_seconds)
                
                return RateLimitResult(
                    allowed=True,
                    limit=rule.requests_per_window,
                    remaining=rule.requests_per_window - current_count - cost,
                    reset_time=datetime.fromtimestamp(window_start + rule.window_size_seconds),
                    algorithm=rule.algorithm,
                    scope=rule.scope,
                    identifier=cache_key,
                    cost=cost
                )
            else:
                # Use in-memory cache
                current_count = self.cache_manager.get(window_key) or 0
                
                if current_count + cost > rule.requests_per_window:
                    return RateLimitResult(
                        allowed=False,
                        limit=rule.requests_per_window,
                        remaining=0,
                        reset_time=datetime.fromtimestamp(window_start + rule.window_size_seconds),
                        retry_after=rule.window_size_seconds - (current_time - window_start),
                        algorithm=rule.algorithm,
                        scope=rule.scope,
                        identifier=cache_key,
                        cost=cost
                    )
                
                # Increment counter
                self.cache_manager.set(window_key, current_count + cost, rule.window_size_seconds)
                
                return RateLimitResult(
                    allowed=True,
                    limit=rule.requests_per_window,
                    remaining=rule.requests_per_window - current_count - cost,
                    reset_time=datetime.fromtimestamp(window_start + rule.window_size_seconds),
                    algorithm=rule.algorithm,
                    scope=rule.scope,
                    identifier=cache_key,
                    cost=cost
                )
        
        except Exception as e:
            self.logger.error(f"Error in fixed window check: {e}")
            raise
    
    async def _check_sliding_window(self, rule: RateLimitRule, cache_key: str, cost: int) -> RateLimitResult:
        """Sliding window rate limiting algorithm"""
        try:
            current_time = int(time.time())
            window_start = current_time - rule.window_size_seconds
            
            if self.redis_connected:
                # Use Redis with sorted sets
                pipe = self.redis_client.pipeline()
                
                # Remove old entries
                pipe.zremrangebyscore(cache_key, 0, window_start)
                
                # Count current entries
                pipe.zcard(cache_key)
                
                # Add current request
                pipe.zadd(cache_key, {str(current_time): current_time})
                
                # Set expiration
                pipe.expire(cache_key, rule.window_size_seconds)
                
                results = await pipe.execute()
                current_count = results[1]
                
                if current_count + cost > rule.requests_per_window:
                    return RateLimitResult(
                        allowed=False,
                        limit=rule.requests_per_window,
                        remaining=0,
                        reset_time=datetime.fromtimestamp(current_time + rule.window_size_seconds),
                        retry_after=rule.window_size_seconds,
                        algorithm=rule.algorithm,
                        scope=rule.scope,
                        identifier=cache_key,
                        cost=cost
                    )
                
                return RateLimitResult(
                    allowed=True,
                    limit=rule.requests_per_window,
                    remaining=rule.requests_per_window - current_count - cost,
                    reset_time=datetime.fromtimestamp(current_time + rule.window_size_seconds),
                    algorithm=rule.algorithm,
                    scope=rule.scope,
                    identifier=cache_key,
                    cost=cost
                )
            else:
                # Simplified sliding window for in-memory
                return await self._check_fixed_window(rule, cache_key, cost)
        
        except Exception as e:
            self.logger.error(f"Error in sliding window check: {e}")
            raise
    
    async def _check_token_bucket(self, rule: RateLimitRule, cache_key: str, cost: int) -> RateLimitResult:
        """Token bucket rate limiting algorithm"""
        try:
            current_time = time.time()
            bucket_key = f"{cache_key}:bucket"
            
            if self.redis_connected:
                # Use Redis for token bucket
                pipe = self.redis_client.pipeline()
                
                # Get current bucket state
                pipe.hmget(bucket_key, "tokens", "last_refill")
                
                results = await pipe.execute()
                bucket_data = results[0]
                
                if bucket_data[0] is None:
                    # Initialize bucket
                    tokens = rule.requests_per_window
                    last_refill = current_time
                else:
                    tokens = float(bucket_data[0])
                    last_refill = float(bucket_data[1])
                
                # Refill tokens
                time_passed = current_time - last_refill
                tokens_to_add = time_passed * (rule.requests_per_window / rule.window_size_seconds)
                tokens = min(rule.requests_per_window, tokens + tokens_to_add)
                
                if tokens < cost:
                    return RateLimitResult(
                        allowed=False,
                        limit=rule.requests_per_window,
                        remaining=int(tokens),
                        reset_time=datetime.fromtimestamp(current_time + (cost - tokens) / (rule.requests_per_window / rule.window_size_seconds)),
                        retry_after=int((cost - tokens) / (rule.requests_per_window / rule.window_size_seconds)),
                        algorithm=rule.algorithm,
                        scope=rule.scope,
                        identifier=cache_key,
                        cost=cost
                    )
                
                # Consume tokens
                tokens -= cost
                
                # Update bucket
                await self.redis_client.hmset(bucket_key, {
                    "tokens": tokens,
                    "last_refill": current_time
                })
                await self.redis_client.expire(bucket_key, rule.window_size_seconds)
                
                return RateLimitResult(
                    allowed=True,
                    limit=rule.requests_per_window,
                    remaining=int(tokens),
                    reset_time=datetime.fromtimestamp(current_time + rule.window_size_seconds),
                    algorithm=rule.algorithm,
                    scope=rule.scope,
                    identifier=cache_key,
                    cost=cost
                )
            else:
                # Simplified token bucket for in-memory
                return await self._check_fixed_window(rule, cache_key, cost)
        
        except Exception as e:
            self.logger.error(f"Error in token bucket check: {e}")
            raise
    
    async def _check_leaky_bucket(self, rule: RateLimitRule, cache_key: str, cost: int) -> RateLimitResult:
        """Leaky bucket rate limiting algorithm"""
        # Simplified implementation - similar to token bucket
        return await self._check_token_bucket(rule, cache_key, cost)
    
    async def _check_adaptive(self, rule: RateLimitRule, cache_key: str, cost: int) -> RateLimitResult:
        """Adaptive rate limiting algorithm"""
        try:
            # Get recent request history
            history_key = f"{cache_key}:history"
            
            if self.redis_connected:
                # Get recent requests (last hour)
                recent_requests = await self.redis_client.lrange(history_key, 0, -1)
                recent_requests = [float(req) for req in recent_requests]
                
                # Calculate adaptive limit based on recent activity
                current_time = time.time()
                hour_ago = current_time - 3600
                
                # Count requests in last hour
                recent_count = len([req for req in recent_requests if req > hour_ago])
                
                # Adaptive logic: reduce limit if high activity
                if recent_count > rule.requests_per_window * 0.8:
                    adaptive_limit = int(rule.requests_per_window * 0.5)
                elif recent_count > rule.requests_per_window * 0.5:
                    adaptive_limit = int(rule.requests_per_window * 0.7)
                else:
                    adaptive_limit = rule.requests_per_window
                
                # Check against adaptive limit
                current_count = await self.redis_client.get(cache_key)
                current_count = int(current_count) if current_count else 0
                
                if current_count + cost > adaptive_limit:
                    return RateLimitResult(
                        allowed=False,
                        limit=adaptive_limit,
                        remaining=0,
                        reset_time=datetime.fromtimestamp(current_time + rule.window_size_seconds),
                        retry_after=rule.window_size_seconds,
                        algorithm=rule.algorithm,
                        scope=rule.scope,
                        identifier=cache_key,
                        cost=cost
                    )
                
                # Update counters
                await self.redis_client.incrby(cache_key, cost)
                await self.redis_client.expire(cache_key, rule.window_size_seconds)
                
                # Add to history
                await self.redis_client.lpush(history_key, current_time)
                await self.redis_client.ltrim(history_key, 0, 1000)  # Keep last 1000 requests
                await self.redis_client.expire(history_key, 3600)  # 1 hour
                
                return RateLimitResult(
                    allowed=True,
                    limit=adaptive_limit,
                    remaining=adaptive_limit - current_count - cost,
                    reset_time=datetime.fromtimestamp(current_time + rule.window_size_seconds),
                    algorithm=rule.algorithm,
                    scope=rule.scope,
                    identifier=cache_key,
                    cost=cost
                )
            else:
                # Fallback to fixed window
                return await self._check_fixed_window(rule, cache_key, cost)
        
        except Exception as e:
            self.logger.error(f"Error in adaptive check: {e}")
            raise
    
    def _generate_cache_key(self, rule: RateLimitRule, identifier: str) -> str:
        """Generate cache key for rate limiting"""
        key_parts = ["rate_limit", rule.name, rule.scope.value, identifier]
        return ":".join(key_parts)
    
    async def get_rate_limit_status(self, identifier: str, scope: RateLimitScope = RateLimitScope.IP) -> Dict[str, Any]:
        """Get current rate limit status for identifier"""
        try:
            status = {
                "identifier": identifier,
                "scope": scope.value,
                "rules": [],
                "overall_status": "allowed"
            }
            
            # Check each applicable rule
            applicable_rules = self._get_applicable_rules(scope)
            
            for rule in applicable_rules:
                cache_key = self._generate_cache_key(rule, identifier)
                
                if self.redis_connected:
                    current_count = await self.redis_client.get(cache_key)
                    current_count = int(current_count) if current_count else 0
                else:
                    current_count = self.cache_manager.get(cache_key) or 0
                
                rule_status = {
                    "name": rule.name,
                    "algorithm": rule.algorithm.value,
                    "limit": rule.requests_per_window,
                    "current": current_count,
                    "remaining": max(0, rule.requests_per_window - current_count),
                    "window_size": rule.window_size_seconds,
                    "enabled": rule.enabled
                }
                
                status["rules"].append(rule_status)
                
                # Check if any rule would block
                if current_count >= rule.requests_per_window:
                    status["overall_status"] = "blocked"
            
            return status
        
        except Exception as e:
            self.logger.error(f"Error getting rate limit status: {e}")
            return {"error": str(e)}
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        try:
            return {
                "total_requests": self.stats["total_requests"],
                "allowed_requests": self.stats["allowed_requests"],
                "blocked_requests": self.stats["blocked_requests"],
                "block_rate": (self.stats["blocked_requests"] / max(self.stats["total_requests"], 1)) * 100,
                "rule_hits": self.stats["rule_hits"],
                "algorithm_usage": self.stats["algorithm_usage"],
                "redis_connected": self.redis_connected,
                "active_rules": len([r for r in self.rules.values() if r.enabled]),
                "total_rules": len(self.rules)
            }
        
        except Exception as e:
            self.logger.error(f"Error getting rate limit stats: {e}")
            return {"error": str(e)}
    
    async def reset_rate_limit(self, identifier: str, scope: RateLimitScope = RateLimitScope.IP) -> bool:
        """Reset rate limit for identifier"""
        try:
            applicable_rules = self._get_applicable_rules(scope)
            
            for rule in applicable_rules:
                cache_key = self._generate_cache_key(rule, identifier)
                
                if self.redis_connected:
                    await self.redis_client.delete(cache_key)
                else:
                    self.cache_manager._evict(cache_key)
            
            self.logger.info(f"Reset rate limit for {scope.value}:{identifier}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error resetting rate limit: {e}")
            return False

# Global rate limiter
_advanced_rate_limiter: Optional[AdvancedRateLimiter] = None

def get_advanced_rate_limiter() -> AdvancedRateLimiter:
    """Get the global advanced rate limiter"""
    global _advanced_rate_limiter
    if _advanced_rate_limiter is None:
        _advanced_rate_limiter = AdvancedRateLimiter()
    return _advanced_rate_limiter

# Rate limiting middleware
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware for FastAPI"""
    try:
        rate_limiter = get_advanced_rate_limiter()
        
        # Extract identifier based on request
        identifier = request.client.host if request.client else "unknown"
        
        # Check rate limit
        result = await rate_limiter.check_rate_limit(
            identifier=identifier,
            scope=RateLimitScope.IP,
            endpoint=request.url.path
        )
        
        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": result.limit,
                    "remaining": result.remaining,
                    "reset_time": result.reset_time.isoformat(),
                    "retry_after": result.retry_after
                },
                headers={
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": str(result.remaining),
                    "X-RateLimit-Reset": str(int(result.reset_time.timestamp())),
                    "Retry-After": str(result.retry_after) if result.retry_after else "60"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_time.timestamp()))
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in rate limit middleware: {e}")
        return await call_next(request)


