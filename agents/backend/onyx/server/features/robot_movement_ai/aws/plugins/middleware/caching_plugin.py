"""
Caching Plugin
==============
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from aws.core.interfaces import MiddlewarePlugin

logger = logging.getLogger(__name__)


class CachingMiddlewarePlugin(MiddlewarePlugin):
    """Redis caching middleware plugin."""
    
    def get_name(self) -> str:
        return "caching"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        middleware_config = config.get("middleware", {})
        return middleware_config.get("enable_caching", True)
    
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup caching middleware."""
        middleware_config = config.get("middleware", {})
        redis_url = middleware_config.get("redis_url")
        cache_ttl = middleware_config.get("cache_ttl", 300)
        
        if not redis_url:
            logger.warning("Redis URL not configured. Caching disabled.")
            return app
        
        class CachingMiddleware(BaseHTTPMiddleware):
            def __init__(self, app):
                super().__init__(app)
                self.redis_url = redis_url
                self.cache_ttl = cache_ttl
                self.redis_client = None
            
            async def dispatch(self, request: Request, call_next):
                # Only cache GET requests
                if request.method != "GET":
                    return await call_next(request)
                
                # Skip certain paths
                skip_paths = ["/health", "/metrics", "/docs", "/openapi.json"]
                if any(request.url.path.startswith(path) for path in skip_paths):
                    return await call_next(request)
                
                # Initialize Redis if needed
                if self.redis_client is None:
                    try:
                        import redis.asyncio as aioredis
                        self.redis_client = await aioredis.from_url(
                            self.redis_url,
                            encoding="utf-8",
                            decode_responses=True
                        )
                    except Exception as e:
                        logger.warning(f"Failed to connect to Redis: {e}")
                        return await call_next(request)
                
                # Generate cache key
                cache_key = f"cache:{request.url.path}:{str(request.query_params)}"
                
                try:
                    # Try cache
                    cached_response = await self.redis_client.get(cache_key)
                    if cached_response:
                        return Response(
                            content=cached_response,
                            media_type="application/json",
                            headers={"X-Cache": "HIT"}
                        )
                    
                    # Cache miss
                    response = await call_next(request)
                    
                    if response.status_code == 200:
                        try:
                            body = b""
                            async for chunk in response.body_iterator:
                                body += chunk
                            
                            await self.redis_client.setex(
                                cache_key,
                                self.cache_ttl,
                                body.decode("utf-8")
                            )
                            
                            return Response(
                                content=body,
                                status_code=response.status_code,
                                headers={**dict(response.headers), "X-Cache": "MISS"}
                            )
                        except Exception as e:
                            logger.warning(f"Failed to cache response: {e}")
                            return response
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Cache error: {e}")
                    return await call_next(request)
        
        app.add_middleware(CachingMiddleware)
        logger.info("Caching enabled")
        
        return app















