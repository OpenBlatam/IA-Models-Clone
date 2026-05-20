"""
Cache Middleware
================

Advanced caching middleware with multiple backends and strategies.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import json
import hashlib
import pickle

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...shared.config import get_settings
from ...infrastructure.external.cache_service import CacheService, CacheConfig


logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategies"""
    NO_CACHE = "no_cache"
    CACHE_FIRST = "cache_first"
    CACHE_ONLY = "cache_only"
    NETWORK_FIRST = "network_first"
    STALE_WHILE_REVALIDATE = "stale_while_revalidate"


class CacheKeyGenerator:
    """Cache key generator"""
    
    @staticmethod
    def generate_key(
        method: str,
        path: str,
        query_params: Dict[str, Any],
        user_id: Optional[str] = None,
        custom_key: Optional[str] = None
    ) -> str:
        """Generate cache key for request"""
        if custom_key:
            return f"custom:{custom_key}"
        
        # Create key components
        components = [method.lower(), path]
        
        # Add query parameters
        if query_params:
            sorted_params = sorted(query_params.items())
            param_str = "&".join(f"{k}={v}" for k, v in sorted_params)
            components.append(param_str)
        
        # Add user ID if provided
        if user_id:
            components.append(f"user:{user_id}")
        
        # Join components and hash
        key_string = ":".join(components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @staticmethod
    def generate_pattern(pattern: str) -> str:
        """Generate cache key pattern"""
        return f"pattern:{pattern}"


@dataclass
class CacheRule:
    """Cache rule configuration"""
    path_pattern: str
    method: str = "GET"
    ttl: int = 300
    strategy: CacheStrategy = CacheStrategy.CACHE_FIRST
    vary_by_user: bool = False
    vary_by_headers: List[str] = None
    custom_key: Optional[str] = None
    enabled: bool = True


class CacheMiddleware:
    """
    Advanced caching middleware
    
    Provides intelligent caching with multiple strategies,
    rule-based configuration, and performance optimization.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.settings = get_settings()
        self.cache_service = cache_service or CacheService(
            CacheConfig(
                backend=self.settings.cache_backend.value,
                default_ttl=self.settings.cache_default_ttl,
                max_size=self.settings.cache_max_size
            )
        )
        self._rules: List[CacheRule] = []
        self._statistics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_sets": 0,
            "cache_invalidations": 0,
            "by_strategy": {strategy.value: 0 for strategy in CacheStrategy}
        }
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default cache rules"""
        # Cache API responses
        self._rules.extend([
            CacheRule(
                path_pattern="/api/v3/workflows",
                method="GET",
                ttl=300,
                strategy=CacheStrategy.CACHE_FIRST,
                vary_by_user=True
            ),
            CacheRule(
                path_pattern="/api/v3/workflows/*",
                method="GET",
                ttl=600,
                strategy=CacheStrategy.CACHE_FIRST,
                vary_by_user=True
            ),
            CacheRule(
                path_pattern="/health",
                method="GET",
                ttl=60,
                strategy=CacheStrategy.CACHE_FIRST
            ),
            CacheRule(
                path_pattern="/metrics",
                method="GET",
                ttl=30,
                strategy=CacheStrategy.CACHE_FIRST
            )
        ])
    
    def add_rule(self, rule: CacheRule) -> None:
        """Add cache rule"""
        self._rules.append(rule)
        logger.info(f"Added cache rule for {rule.method} {rule.path_pattern}")
    
    def remove_rule(self, path_pattern: str, method: str = "GET") -> bool:
        """Remove cache rule"""
        for i, rule in enumerate(self._rules):
            if rule.path_pattern == path_pattern and rule.method == method:
                del self._rules[i]
                logger.info(f"Removed cache rule for {method} {path_pattern}")
                return True
        return False
    
    def get_matching_rule(self, path: str, method: str) -> Optional[CacheRule]:
        """Get matching cache rule for path and method"""
        for rule in self._rules:
            if not rule.enabled:
                continue
            
            if rule.method != method:
                continue
            
            # Simple pattern matching (in production, use regex)
            if self._matches_pattern(path, rule.path_pattern):
                return rule
        
        return None
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern"""
        if pattern == path:
            return True
        
        if "*" in pattern:
            # Simple wildcard matching
            pattern_parts = pattern.split("*")
            if len(pattern_parts) == 2:
                prefix, suffix = pattern_parts
                return path.startswith(prefix) and path.endswith(suffix)
        
        return False
    
    async def get_cached_response(self, request: Request) -> Optional[Response]:
        """Get cached response for request"""
        try:
            # Get matching rule
            rule = self.get_matching_rule(request.url.path, request.method)
            if not rule:
                return None
            
            # Generate cache key
            cache_key = self._generate_cache_key(request, rule)
            
            # Get from cache
            cached_data = await self.cache_service.get(cache_key)
            if cached_data:
                self._statistics["cache_hits"] += 1
                self._statistics["by_strategy"][rule.strategy.value] += 1
                
                # Create response from cached data
                return self._create_response_from_cache(cached_data)
            
            self._statistics["cache_misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached response: {e}")
            return None
    
    async def cache_response(
        self,
        request: Request,
        response: Response,
        rule: CacheRule
    ) -> None:
        """Cache response"""
        try:
            # Only cache successful responses
            if response.status_code not in [200, 201, 202]:
                return
            
            # Generate cache key
            cache_key = self._generate_cache_key(request, rule)
            
            # Prepare cache data
            cache_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.body.decode() if response.body else "",
                "cached_at": datetime.utcnow().isoformat()
            }
            
            # Cache the response
            await self.cache_service.set(cache_key, cache_data, rule.ttl)
            self._statistics["cache_sets"] += 1
            
            logger.debug(f"Cached response for {request.method} {request.url.path}")
            
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
    
    async def invalidate_cache(
        self,
        pattern: str,
        method: Optional[str] = None
    ) -> int:
        """Invalidate cache entries matching pattern"""
        try:
            # Generate pattern key
            pattern_key = CacheKeyGenerator.generate_pattern(pattern)
            
            # Get all keys matching pattern
            keys = await self.cache_service.keys(pattern)
            
            # Filter by method if specified
            if method:
                keys = [key for key in keys if key.startswith(f"{method.lower()}:")]
            
            # Delete matching keys
            count = 0
            for key in keys:
                if await self.cache_service.delete(key):
                    count += 1
            
            self._statistics["cache_invalidations"] += count
            logger.info(f"Invalidated {count} cache entries for pattern {pattern}")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return 0
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cache entries for a specific user"""
        try:
            # Get all keys
            all_keys = await self.cache_service.keys("*")
            
            # Filter keys for user
            user_keys = [key for key in all_keys if f"user:{user_id}" in key]
            
            # Delete user keys
            count = 0
            for key in user_keys:
                if await self.cache_service.delete(key):
                    count += 1
            
            self._statistics["cache_invalidations"] += count
            logger.info(f"Invalidated {count} cache entries for user {user_id}")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to invalidate user cache: {e}")
            return 0
    
    def _generate_cache_key(self, request: Request, rule: CacheRule) -> str:
        """Generate cache key for request and rule"""
        # Extract query parameters
        query_params = dict(request.query_params)
        
        # Get user ID if needed
        user_id = None
        if rule.vary_by_user:
            # In a real implementation, extract user ID from request
            user_id = "user_123"  # Mock user ID
        
        # Generate key
        return CacheKeyGenerator.generate_key(
            method=request.method,
            path=request.url.path,
            query_params=query_params,
            user_id=user_id,
            custom_key=rule.custom_key
        )
    
    def _create_response_from_cache(self, cache_data: Dict[str, Any]) -> Response:
        """Create response from cached data"""
        response = Response(
            content=cache_data["body"],
            status_code=cache_data["status_code"],
            headers=cache_data["headers"]
        )
        
        # Add cache headers
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Cached-At"] = cache_data["cached_at"]
        
        return response
    
    async def process_request(self, request: Request) -> Optional[Response]:
        """Process request through cache middleware"""
        try:
            # Get matching rule
            rule = self.get_matching_rule(request.url.path, request.method)
            if not rule:
                return None
            
            # Apply cache strategy
            if rule.strategy == CacheStrategy.NO_CACHE:
                return None
            
            if rule.strategy in [CacheStrategy.CACHE_FIRST, CacheStrategy.CACHE_ONLY]:
                cached_response = await self.get_cached_response(request)
                if cached_response:
                    return cached_response
                
                if rule.strategy == CacheStrategy.CACHE_ONLY:
                    # Return 503 Service Unavailable if cache only and no cache
                    return Response(
                        content="Service temporarily unavailable",
                        status_code=503,
                        headers={"X-Cache": "MISS"}
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Cache middleware request processing failed: {e}")
            return None
    
    async def process_response(
        self,
        request: Request,
        response: Response
    ) -> Response:
        """Process response through cache middleware"""
        try:
            # Get matching rule
            rule = self.get_matching_rule(request.url.path, request.method)
            if not rule:
                return response
            
            # Apply cache strategy
            if rule.strategy in [CacheStrategy.CACHE_FIRST, CacheStrategy.NETWORK_FIRST]:
                await self.cache_response(request, response, rule)
            
            # Add cache headers
            response.headers["X-Cache-Strategy"] = rule.strategy.value
            response.headers["X-Cache-TTL"] = str(rule.ttl)
            
            return response
            
        except Exception as e:
            logger.error(f"Cache middleware response processing failed: {e}")
            return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache middleware statistics"""
        total_requests = self._statistics["cache_hits"] + self._statistics["cache_misses"]
        hit_rate = (self._statistics["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self._statistics,
            "hit_rate": hit_rate,
            "rules_count": len(self._rules),
            "cache_service_stats": self.cache_service.get_statistics()
        }
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get cache rules"""
        return [
            {
                "path_pattern": rule.path_pattern,
                "method": rule.method,
                "ttl": rule.ttl,
                "strategy": rule.strategy.value,
                "vary_by_user": rule.vary_by_user,
                "vary_by_headers": rule.vary_by_headers,
                "custom_key": rule.custom_key,
                "enabled": rule.enabled
            }
            for rule in self._rules
        ]


# Global cache middleware instance
_cache_middleware: Optional[CacheMiddleware] = None


def get_cache_middleware() -> CacheMiddleware:
    """Get global cache middleware instance"""
    global _cache_middleware
    if _cache_middleware is None:
        _cache_middleware = CacheMiddleware()
    return _cache_middleware


# FastAPI dependency
async def get_cache_middleware_dependency() -> CacheMiddleware:
    """FastAPI dependency for cache middleware"""
    return get_cache_middleware()




