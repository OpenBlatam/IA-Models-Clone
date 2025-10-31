"""
Serverless Optimization Utilities
Minimizes cold start times and optimizes FastAPI for serverless environments
(AWS Lambda, Azure Functions, Google Cloud Functions)
"""

import os
import sys
import importlib
import logging
from typing import Callable, Any, Optional
from functools import lru_cache, wraps
import time

logger = logging.getLogger(__name__)


# Detect serverless environment
def is_serverless() -> bool:
    """Detect if running in serverless environment"""
    return any([
        os.getenv("AWS_LAMBDA_FUNCTION_NAME"),  # AWS Lambda
        os.getenv("FUNCTIONS_WORKER_RUNTIME"),  # Azure Functions
        os.getenv("K_SERVICE"),  # Cloud Run / Knative
        os.getenv("_HANDLER"),  # Generic serverless
    ])


# Optimize imports for faster cold starts
def lazy_import(module_name: str):
    """Lazy import to reduce cold start time"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if module_name not in sys.modules:
                importlib.import_module(module_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Cache expensive operations
class LRUCache:
    """LRU Cache for frequently accessed data"""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache = {}
        self._access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self._cache:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        if len(self._cache) >= self.maxsize:
            # Remove least recently used
            if self._access_order:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
        
        self._cache[key] = value
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        self._cache.clear()
        self._access_order.clear()


# Global cache instance
_lru_cache = LRUCache(maxsize=256)


@lru_cache(maxsize=128)
def get_cached_config(key: str, default: Any = None) -> Any:
    """Get cached configuration value"""
    return os.getenv(key, default)


# Optimize JSON serialization
try:
    import orjson
    JSON_ENCODER = orjson
    HAS_ORJSON = True
except ImportError:
    import json
    JSON_ENCODER = json
    HAS_ORJSON = False


def fast_json_dumps(obj: Any) -> str:
    """Fast JSON serialization"""
    if HAS_ORJSON:
        return orjson.dumps(obj).decode('utf-8')
    else:
        return json.dumps(obj)


def fast_json_loads(s: str) -> Any:
    """Fast JSON deserialization"""
    if HAS_ORJSON:
        return orjson.loads(s)
    else:
        return json.loads(s)


# Connection pooling for serverless
class ConnectionPool:
    """
    Lightweight connection pool for serverless environments
    Reuses connections within the same execution context
    """
    
    def __init__(self):
        self._pools = {}
    
    def get_pool(self, pool_type: str, factory: Callable):
        """Get or create connection pool"""
        if pool_type not in self._pools:
            self._pools[pool_type] = factory()
        return self._pools[pool_type]
    
    def clear(self):
        """Clear all pools"""
        for pool in self._pools.values():
            if hasattr(pool, 'close'):
                pool.close()
        self._pools.clear()


_connection_pool = ConnectionPool()


# Warm-up function for serverless
def warm_up():
    """
    Warm-up function to reduce cold start time
    Pre-loads commonly used modules and initializes connections
    """
    start_time = time.time()
    
    try:
        # Pre-import commonly used modules
        import json
        import logging
        import time
        import uuid
        
        # Pre-initialize connections (lightweight)
        # Don't create heavy connections during warm-up
        
        warmup_time = time.time() - start_time
        logger.info(f"Warm-up completed in {warmup_time:.3f}s")
        return warmup_time
    except Exception as e:
        logger.error(f"Warm-up failed: {e}")
        return None


# FastAPI app factory optimized for serverless
def create_serverless_app():
    """
    Create FastAPI app optimized for serverless
    Minimizes initialization time
    """
    from fastapi import FastAPI
    from config import settings
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        # Disable docs in production for faster startup
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
    )
    
    return app


# Lazy middleware loading
class LazyMiddleware:
    """Lazy-load middleware to reduce cold start"""
    
    def __init__(self, middleware_class, *args, **kwargs):
        self.middleware_class = middleware_class
        self.args = args
        self.kwargs = kwargs
        self._instance = None
    
    def get_instance(self):
        """Get middleware instance (lazy loaded)"""
        if self._instance is None:
            self._instance = self.middleware_class(*self.args, **self.kwargs)
        return self._instance


# Optimize for AWS Lambda
def lambda_handler(event, context):
    """
    AWS Lambda handler wrapper
    Converts Lambda event to ASGI request
    """
    try:
        from mangum import Mangum
        from app import app
        
        handler = Mangum(app)
        return handler(event, context)
    except ImportError:
        logger.error("mangum not installed. Install with: pip install mangum")
        raise


# Optimize for Azure Functions
def azure_function_handler(req):
    """
    Azure Functions handler wrapper
    """
    try:
        from azure.functions import HttpRequest, HttpResponse
        from fastapi import Request
        import asyncio
        
        # Convert Azure request to FastAPI request
        # This is a simplified version
        # In production, use proper ASGI adapter
        
        logger.warning("Azure Functions adapter not fully implemented")
        return {"status": "not implemented"}
    except Exception as e:
        logger.error(f"Azure Functions handler error: {e}")
        raise


# Performance monitoring for serverless
class ServerlessMetrics:
    """Metrics specific to serverless environments"""
    
    def __init__(self):
        self.request_times = []
        self.cold_starts = 0
        self.warm_starts = 0
    
    def record_request(self, duration: float, is_cold_start: bool):
        """Record request metrics"""
        self.request_times.append(duration)
        if is_cold_start:
            self.cold_starts += 1
        else:
            self.warm_starts += 1
    
    def get_stats(self) -> dict:
        """Get serverless metrics"""
        if not self.request_times:
            return {}
        
        return {
            "avg_response_time": sum(self.request_times) / len(self.request_times),
            "max_response_time": max(self.request_times),
            "min_response_time": min(self.request_times),
            "total_requests": len(self.request_times),
            "cold_starts": self.cold_starts,
            "warm_starts": self.warm_starts,
            "cold_start_rate": self.cold_starts / len(self.request_times) if self.request_times else 0
        }


_serverless_metrics = ServerlessMetrics()


def optimize_for_serverless():
    """
    Apply serverless optimizations
    Should be called at application startup
    """
    if not is_serverless():
        logger.info("Not in serverless environment, skipping optimizations")
        return
    
    logger.info("Applying serverless optimizations...")
    
    # Set environment variables for better performance
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    
    # Optimize garbage collection
    import gc
    gc.set_threshold(700, 10, 10)
    
    # Pre-warm connections (lightweight)
    warm_up()
    
    logger.info("Serverless optimizations applied")






