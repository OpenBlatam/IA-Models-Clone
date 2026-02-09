"""
Advanced rate limiting and throttling for Blaze AI.

This module provides sophisticated rate limiting with multiple algorithms,
user-based limits, adaptive throttling, and distributed rate limiting support.
"""

import asyncio
import time
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict, deque
import redis.asyncio as redis

# =============================================================================
# Types
# =============================================================================

RateLimitKey = Union[str, int]
RateLimitValue = Union[int, float]

# =============================================================================
# Enums
# =============================================================================

class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    LEAKY_BUCKET = "leaky_bucket"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"

class ThrottleAction(Enum):
    """Actions to take when rate limit is exceeded."""
    REJECT = "reject"
    QUEUE = "queue"
    DELAY = "delay"
    DEGRADE = "degrade"

# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 50
    window_size: int = 60  # seconds
    action: ThrottleAction = ThrottleAction.REJECT
    enable_user_limits: bool = True
    enable_ip_limits: bool = True
    enable_global_limits: bool = True
    redis_url: Optional[str] = None
    enable_distributed: bool = False

@dataclass
class UserRateLimit:
    """User-specific rate limit configuration."""
    user_id: str
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 50
    priority: int = 1  # Higher priority = higher limits

@dataclass
class ThrottleConfig:
    """Throttling configuration."""
    max_queue_size: int = 1000
    max_wait_time: float = 30.0  # seconds
    enable_priority_queue: bool = True
    enable_adaptive_throttling: bool = True

# =============================================================================
# Base Rate Limiter
# =============================================================================

class BaseRateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self._setup_redis()
    
    def _setup_redis(self):
        """Setup Redis client if distributed mode is enabled."""
        if self.config.enable_distributed and self.config.redis_url:
            try:
                self.redis_client = redis.from_url(self.config.redis_url)
            except Exception:
                # Fallback to local mode if Redis is unavailable
                self.redis_client = None
    
    @abstractmethod
    async def is_allowed(self, key: RateLimitKey) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed."""
        pass
    
    @abstractmethod
    async def record_request(self, key: RateLimitKey, timestamp: Optional[float] = None):
        """Record a request."""
        pass
    
    async def get_usage(self, key: RateLimitKey) -> Dict[str, Any]:
        """Get current usage statistics."""
        pass

# =============================================================================
# Fixed Window Rate Limiter
# =============================================================================

class FixedWindowRateLimiter(BaseRateLimiter):
    """Fixed window rate limiter implementation."""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.windows: Dict[RateLimitKey, Dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: RateLimitKey) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed in fixed window."""
        current_time = time.time()
        window_start = int(current_time // self.config.window_size) * self.config.window_size
        
        async with self._lock:
            if key not in self.windows or self.windows[key].get('window_start', 0) != window_start:
                # New window, reset counter
                self.windows[key] = {
                    'window_start': window_start,
                    'count': 0,
                    'last_request': current_time
                }
            
            current_count = self.windows[key]['count']
            is_allowed = current_count < self.config.requests_per_minute
            
            return is_allowed, {
                'current_count': current_count,
                'limit': self.config.requests_per_minute,
                'window_start': window_start,
                'window_end': window_start + self.config.window_size,
                'reset_time': window_start + self.config.window_size
            }
    
    async def record_request(self, key: RateLimitKey, timestamp: Optional[float] = None):
        """Record a request in the current window."""
        current_time = timestamp or time.time()
        window_start = int(current_time // self.config.window_size) * self.config.window_size
        
        async with self._lock:
            if key not in self.windows or self.windows[key].get('window_start', 0) != window_start:
                self.windows[key] = {
                    'window_start': window_start,
                    'count': 0,
                    'last_request': current_time
                }
            
            self.windows[key]['count'] += 1
            self.windows[key]['last_request'] = current_time

# =============================================================================
# Sliding Window Rate Limiter
# =============================================================================

class SlidingWindowRateLimiter(BaseRateLimiter):
    """Sliding window rate limiter implementation."""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.requests: Dict[RateLimitKey, deque] = defaultdict(lambda: deque())
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: RateLimitKey) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed in sliding window."""
        current_time = time.time()
        window_start = current_time - self.config.window_size
        
        async with self._lock:
            # Remove expired requests
            while (self.requests[key] and 
                   self.requests[key][0] < window_start):
                self.requests[key].popleft()
            
            current_count = len(self.requests[key])
            is_allowed = current_count < self.config.requests_per_minute
            
            return is_allowed, {
                'current_count': current_count,
                'limit': self.config.requests_per_minute,
                'window_start': window_start,
                'window_end': current_time,
                'reset_time': current_time + self.config.window_size
            }
    
    async def record_request(self, key: RateLimitKey, timestamp: Optional[float] = None):
        """Record a request in the sliding window."""
        current_time = timestamp or time.time()
        
        async with self._lock:
            self.requests[key].append(current_time)

# =============================================================================
# Token Bucket Rate Limiter
# =============================================================================

class TokenBucketRateLimiter(BaseRateLimiter):
    """Token bucket rate limiter implementation."""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.buckets: Dict[RateLimitKey, Dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: RateLimitKey) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed in token bucket."""
        current_time = time.time()
        
        async with self._lock:
            if key not in self.buckets:
                self.buckets[key] = {
                    'tokens': self.config.burst_limit,
                    'last_refill': current_time,
                    'refill_rate': self.config.requests_per_minute / 60.0
                }
            
            bucket = self.buckets[key]
            
            # Refill tokens
            time_passed = current_time - bucket['last_refill']
            tokens_to_add = time_passed * bucket['refill_rate']
            bucket['tokens'] = min(
                self.config.burst_limit,
                bucket['tokens'] + tokens_to_add
            )
            bucket['last_refill'] = current_time
            
            is_allowed = bucket['tokens'] >= 1
            
            return is_allowed, {
                'current_tokens': bucket['tokens'],
                'max_tokens': self.config.burst_limit,
                'refill_rate': bucket['refill_rate'],
                'next_refill': current_time + (1 / bucket['refill_rate'])
            }
    
    async def record_request(self, key: RateLimitKey, timestamp: Optional[float] = None):
        """Consume a token from the bucket."""
        async with self._lock:
            if key in self.buckets:
                self.buckets[key]['tokens'] = max(0, self.buckets[key]['tokens'] - 1)

# =============================================================================
# Adaptive Rate Limiter
# =============================================================================

class AdaptiveRateLimiter(BaseRateLimiter):
    """Adaptive rate limiter that adjusts limits based on system performance."""
    
    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.base_limiter = SlidingWindowRateLimiter(config)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.adaptive_multiplier = 1.0
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: RateLimitKey) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed with adaptive limits."""
        # Get base rate limit result
        is_allowed, base_info = await self.base_limiter.is_allowed(key)
        
        # Apply adaptive multiplier
        adaptive_limit = int(self.config.requests_per_minute * self.adaptive_multiplier)
        adaptive_allowed = base_info['current_count'] < adaptive_limit
        
        return adaptive_allowed, {
            **base_info,
            'adaptive_limit': adaptive_limit,
            'adaptive_multiplier': self.adaptive_multiplier,
            'base_limit': self.config.requests_per_minute
        }
    
    async def record_request(self, key: RateLimitKey, timestamp: Optional[float] = None):
        """Record a request and update performance metrics."""
        await self.base_limiter.record_request(key, timestamp)
    
    async def update_performance_metric(self, metric_name: str, value: float):
        """Update performance metric and adjust adaptive multiplier."""
        async with self._lock:
            self.performance_metrics[metric_name].append(value)
            
            # Keep only recent metrics
            if len(self.performance_metrics[metric_name]) > 100:
                self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-100:]
            
            # Calculate adaptive multiplier based on performance
            if metric_name == 'response_time':
                avg_response_time = sum(self.performance_metrics[metric_name]) / len(self.performance_metrics[metric_name])
                target_response_time = 1.0  # 1 second target
                
                if avg_response_time > target_response_time * 1.5:
                    # System is slow, reduce rate limit
                    self.adaptive_multiplier = max(0.5, self.adaptive_multiplier * 0.9)
                elif avg_response_time < target_response_time * 0.5:
                    # System is fast, increase rate limit
                    self.adaptive_multiplier = min(2.0, self.adaptive_multiplier * 1.1)

# =============================================================================
# Rate Limiter Factory
# =============================================================================

class RateLimiterFactory:
    """Factory for creating rate limiters."""
    
    @staticmethod
    def create_limiter(config: RateLimitConfig) -> BaseRateLimiter:
        """Create a rate limiter based on configuration."""
        if config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return FixedWindowRateLimiter(config)
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return SlidingWindowRateLimiter(config)
        elif config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return TokenBucketRateLimiter(config)
        elif config.algorithm == RateLimitAlgorithm.ADAPTIVE:
            return AdaptiveRateLimiter(config)
        else:
            raise ValueError(f"Unsupported rate limiting algorithm: {config.algorithm}")

# =============================================================================
# Rate Limit Manager
# =============================================================================

class RateLimitManager:
    """Manages multiple rate limiters for different contexts."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.limiters: Dict[str, BaseRateLimiter] = {}
        self.user_limits: Dict[str, UserRateLimit] = {}
        self.throttle_config = ThrottleConfig()
        self._setup_limiters()
    
    def _setup_limiters(self):
        """Setup rate limiters for different contexts."""
        if self.config.enable_global_limits:
            self.limiters['global'] = RateLimiterFactory.create_limiter(self.config)
        
        if self.config.enable_ip_limits:
            ip_config = RateLimitConfig(
                **self.config.__dict__,
                requests_per_minute=self.config.requests_per_minute // 2
            )
            self.limiters['ip'] = RateLimiterFactory.create_limiter(ip_config)
    
    def add_user_limit(self, user_limit: UserRateLimit):
        """Add user-specific rate limit."""
        self.user_limits[user_limit.user_id] = user_limit
        
        # Create user-specific limiter
        user_config = RateLimitConfig(
            **self.config.__dict__,
            requests_per_minute=user_limit.requests_per_minute,
            requests_per_hour=user_limit.requests_per_hour,
            requests_per_day=user_limit.requests_per_day,
            burst_limit=user_limit.burst_limit
        )
        self.limiters[f'user_{user_limit.user_id}'] = RateLimiterFactory.create_limiter(user_config)
    
    async def check_rate_limit(self, 
                              user_id: Optional[str] = None,
                              ip_address: Optional[str] = None,
                              context: str = 'default') -> Tuple[bool, Dict[str, Any]]:
        """Check rate limits for all applicable contexts."""
        results = {}
        all_allowed = True
        
        # Check global limits
        if 'global' in self.limiters:
            allowed, info = await self.limiters['global'].is_allowed(context)
            results['global'] = {'allowed': allowed, 'info': info}
            all_allowed = all_allowed and allowed
        
        # Check IP limits
        if ip_address and 'ip' in self.limiters:
            allowed, info = await self.limiters['ip'].is_allowed(ip_address)
            results['ip'] = {'allowed': allowed, 'info': info}
            all_allowed = all_allowed and allowed
        
        # Check user limits
        if user_id and f'user_{user_id}' in self.limiters:
            allowed, info = await self.limiters[f'user_{user_id}'].is_allowed(context)
            results['user'] = {'allowed': allowed, 'info': info}
            all_allowed = all_allowed and allowed
        
        return all_allowed, results
    
    async def record_request(self, 
                           user_id: Optional[str] = None,
                           ip_address: Optional[str] = None,
                           context: str = 'default'):
        """Record requests for all applicable contexts."""
        # Record in global limiter
        if 'global' in self.limiters:
            await self.limiters['global'].record_request(context)
        
        # Record in IP limiter
        if ip_address and 'ip' in self.limiters:
            await self.limiters['ip'].record_request(ip_address)
        
        # Record in user limiter
        if user_id and f'user_{user_id}' in self.limiters:
            await self.limiters[f'user_{user_id}'].record_request(context)
    
    def get_limiter_status(self) -> Dict[str, Any]:
        """Get status of all rate limiters."""
        status = {}
        for name, limiter in self.limiters.items():
            if hasattr(limiter, 'get_usage'):
                status[name] = limiter.get_usage('default')
            else:
                status[name] = {'type': type(limiter).__name__}
        return status

# =============================================================================
# Throttling Queue
# =============================================================================

class ThrottlingQueue:
    """Queue for handling throttled requests."""
    
    def __init__(self, config: ThrottleConfig):
        self.config = config
        self.queue: deque = deque()
        self.processing = False
        self._lock = asyncio.Lock()
    
    async def add_request(self, request: Dict[str, Any], priority: int = 1) -> bool:
        """Add request to throttling queue."""
        if len(self.queue) >= self.config.max_queue_size:
            return False
        
        queue_item = {
            'request': request,
            'priority': priority,
            'timestamp': time.time(),
            'id': hashlib.md5(f"{time.time()}{request}".encode()).hexdigest()
        }
        
        async with self._lock:
            if self.config.enable_priority_queue:
                # Insert based on priority
                insert_index = 0
                for i, item in enumerate(self.queue):
                    if item['priority'] < priority:
                        insert_index = i + 1
                    else:
                        break
                self.queue.insert(insert_index, queue_item)
            else:
                self.queue.append(queue_item)
        
        return True
    
    async def get_next_request(self) -> Optional[Dict[str, Any]]:
        """Get next request from queue."""
        async with self._lock:
            if self.queue:
                return self.queue.popleft()
        return None
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        async with self._lock:
            return {
                'queue_size': len(self.queue),
                'max_size': self.config.max_queue_size,
                'processing': self.processing,
                'oldest_request': self.queue[0]['timestamp'] if self.queue else None,
                'newest_request': self.queue[-1]['timestamp'] if self.queue else None
            }
