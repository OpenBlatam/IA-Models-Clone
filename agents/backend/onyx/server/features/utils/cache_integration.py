from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
from typing import Any, Optional, Dict, List, Callable, Awaitable, Union, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import aioredis
from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import structlog
from .advanced_caching_system import (
from typing import Any, List, Dict, Optional
"""
🔗 Cache Integration
====================

Easy integration of caching system with existing applications:
- FastAPI integration
- Database integration
- External API integration
- Session management
- Configuration management
- Template caching
- Static asset caching
"""



    AdvancedCachingSystem, CacheConfig, CacheType, CacheLevel,
    cache_result, static_cache, dynamic_cache
)

logger = structlog.get_logger(__name__)

class IntegrationType(Enum):
    """Integration types"""
    FASTAPI = "fastapi"
    DATABASE = "database"
    API = "api"
    SESSION = "session"
    CONFIG = "config"
    TEMPLATE = "template"
    ASSET = "asset"

@dataclass
class CacheIntegrationConfig:
    """Configuration for cache integration"""
    # Cache system settings
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    
    # Integration settings
    enable_fastapi_middleware: bool = True
    enable_database_caching: bool = True
    enable_api_caching: bool = True
    enable_session_caching: bool = True
    
    # Session settings
    session_ttl: int = 3600
    session_prefix: str = "session:"
    
    # API settings
    api_cache_ttl: int = 300
    api_cache_prefix: str = "api:"
    
    # Database settings
    db_cache_ttl: int = 1800
    db_cache_prefix: str = "db:"

class FastAPICacheMiddleware:
    """FastAPI middleware for automatic caching."""
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CacheIntegrationConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
        self.cacheable_routes = set()
        self.cache_headers = {
            "X-Cache": "HIT",
            "X-Cache-TTL": str(config.api_cache_ttl)
        }
    
    def add_cacheable_route(self, path: str, ttl: int = None):
        """Add route to cacheable routes."""
        self.cacheable_routes.add((path, ttl or self.config.api_cache_ttl))
    
    async def __call__(self, request: Request, call_next):
        """Process request with caching."""
        # Check if route is cacheable
        cache_ttl = None
        for path, ttl in self.cacheable_routes:
            if request.url.path.startswith(path):
                cache_ttl = ttl
                break
        
        if cache_ttl is None:
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"{self.config.api_cache_prefix}{request.url.path}:{hash(str(request.query_params))}"
        
        # Try to get from cache
        cached_response = await self.cache_system.get(cache_key, CacheType.DYNAMIC)
        if cached_response is not None:
            return JSONResponse(
                content=cached_response,
                headers=self.cache_headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Cache response if successful
        if response.status_code == 200:
            try:
                response_body = await response.body()
                response_data = response_body.decode()
                await self.cache_system.set(
                    cache_key, response_data, CacheType.DYNAMIC, cache_ttl
                )
            except Exception as e:
                logger.error(f"Cache response error: {e}")
        
        return response

class DatabaseCacheIntegration:
    """Database caching integration."""
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CacheIntegrationConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
    
    async def cached_query(self, query_key: str, query_func: Callable, 
                          ttl: int = None, *args, **kwargs) -> Any:
        """Execute cached database query."""
        cache_key = f"{self.config.db_cache_prefix}{query_key}"
        ttl = ttl or self.config.db_cache_ttl
        
        # Try cache first
        cached_result = await self.cache_system.get(cache_key, CacheType.DYNAMIC)
        if cached_result is not None:
            return cached_result
        
        # Execute query
        result = await query_func(*args, **kwargs)
        
        # Cache result
        if result is not None:
            await self.cache_system.set(cache_key, result, CacheType.DYNAMIC, ttl)
        
        return result
    
    async def invalidate_query_pattern(self, pattern: str) -> None:
        """Invalidate cache entries matching pattern."""
        # This would require Redis SCAN or similar for production
        # For now, we'll use a simple approach
        logger.info(f"Invalidating cache pattern: {pattern}")
    
    def query_cache_decorator(self, ttl: int = None):
        """Decorator for caching database queries."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                # Generate cache key from function name and arguments
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                return await self.cached_query(cache_key, func, ttl, *args, **kwargs)
            return wrapper
        return decorator

class SessionCacheIntegration:
    """Session caching integration."""
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CacheIntegrationConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session from cache."""
        cache_key = f"{self.config.session_prefix}{session_id}"
        return await self.cache_system.get(cache_key, CacheType.DYNAMIC)
    
    async def set_session(self, session_id: str, session_data: Dict[str, Any], 
                         ttl: int = None) -> None:
        """Set session in cache."""
        cache_key = f"{self.config.session_prefix}{session_id}"
        ttl = ttl or self.config.session_ttl
        await self.cache_system.set(cache_key, session_data, CacheType.DYNAMIC, ttl)
    
    async def delete_session(self, session_id: str) -> None:
        """Delete session from cache."""
        cache_key = f"{self.config.session_prefix}{session_id}"
        await self.cache_system.delete(cache_key)
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update session data."""
        session_data = await self.get_session(session_id)
        if session_data:
            session_data.update(updates)
            await self.set_session(session_id, session_data)

class ConfigCacheIntegration:
    """Configuration caching integration."""
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CacheIntegrationConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
    
    async def get_config(self, config_key: str) -> Optional[Any]:
        """Get configuration from cache."""
        cache_key = f"config:{config_key}"
        return await self.cache_system.get(cache_key, CacheType.STATIC)
    
    async def set_config(self, config_key: str, config_value: Any) -> None:
        """Set configuration in cache."""
        cache_key = f"config:{config_key}"
        await self.cache_system.set(cache_key, config_value, CacheType.STATIC)
    
    async def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations from cache."""
        # This would require Redis SCAN in production
        # For now, return common configs
        configs = {}
        common_keys = ["app", "database", "redis", "features"]
        
        for key in common_keys:
            value = await self.get_config(key)
            if value is not None:
                configs[key] = value
        
        return configs
    
    async def reload_configs(self, config_loader: Callable) -> None:
        """Reload all configurations."""
        try:
            configs = await config_loader()
            for key, value in configs.items():
                await self.set_config(key, value)
            logger.info("Configurations reloaded successfully")
        except Exception as e:
            logger.error(f"Config reload error: {e}")

class TemplateCacheIntegration:
    """Template caching integration."""
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CacheIntegrationConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
    
    async def get_template(self, template_name: str, template_data: Dict[str, Any] = None) -> Optional[str]:
        """Get rendered template from cache."""
        cache_key = f"template:{template_name}:{hash(str(template_data or {}))}"
        return await self.cache_system.get(cache_key, CacheType.STATIC)
    
    async def set_template(self, template_name: str, rendered_template: str, 
                          template_data: Dict[str, Any] = None) -> None:
        """Set rendered template in cache."""
        cache_key = f"template:{template_name}:{hash(str(template_data or {}))}"
        await self.cache_system.set(cache_key, rendered_template, CacheType.STATIC)
    
    async def invalidate_template(self, template_name: str) -> None:
        """Invalidate template cache."""
        # This would require pattern-based invalidation in production
        logger.info(f"Invalidating template cache: {template_name}")
    
    async def render_template_cached(self, template_name: str, template_data: Dict[str, Any], 
                                   render_func: Callable) -> str:
        """Render template with caching."""
        # Try cache first
        cached_template = await self.get_template(template_name, template_data)
        if cached_template is not None:
            return cached_template
        
        # Render template
        rendered_template = await render_func(template_name, template_data)
        
        # Cache result
        await self.set_template(template_name, rendered_template, template_data)
        
        return rendered_template

class AssetCacheIntegration:
    """Static asset caching integration."""
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CacheIntegrationConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
    
    async def get_asset(self, asset_path: str) -> Optional[bytes]:
        """Get asset from cache."""
        cache_key = f"asset:{asset_path}"
        return await self.cache_system.get(cache_key, CacheType.STATIC)
    
    async def set_asset(self, asset_path: str, asset_data: bytes) -> None:
        """Set asset in cache."""
        cache_key = f"asset:{asset_path}"
        await self.cache_system.set(cache_key, asset_data, CacheType.STATIC)
    
    async def get_asset_cached(self, asset_path: str, loader_func: Callable) -> bytes:
        """Get asset with caching."""
        # Try cache first
        cached_asset = await self.get_asset(asset_path)
        if cached_asset is not None:
            return cached_asset
        
        # Load asset
        asset_data = await loader_func(asset_path)
        
        # Cache result
        await self.set_asset(asset_path, asset_data)
        
        return asset_data

class CacheIntegrationManager:
    """Main cache integration manager."""
    
    def __init__(self, config: CacheIntegrationConfig):
        
    """__init__ function."""
self.config = config
        self.cache_system = AdvancedCachingSystem(config.cache_config)
        
        # Initialize integrations
        self.fastapi_middleware = FastAPICacheMiddleware(self.cache_system, config)
        self.database_cache = DatabaseCacheIntegration(self.cache_system, config)
        self.session_cache = SessionCacheIntegration(self.cache_system, config)
        self.config_cache = ConfigCacheIntegration(self.cache_system, config)
        self.template_cache = TemplateCacheIntegration(self.cache_system, config)
        self.asset_cache = AssetCacheIntegration(self.cache_system, config)
    
    async def initialize(self) -> Any:
        """Initialize cache integration manager."""
        await self.cache_system.initialize()
        
        # Warm up with common configurations
        await self._warmup_configs()
        
        logger.info("Cache integration manager initialized")
    
    async def shutdown(self) -> Any:
        """Shutdown cache integration manager."""
        await self.cache_system.shutdown()
        logger.info("Cache integration manager shutdown complete")
    
    async def _warmup_configs(self) -> Any:
        """Warm up with common configurations."""
        common_configs = {
            "app": {
                "name": "Blatam Academy",
                "version": "1.0.0",
                "environment": "production"
            },
            "features": {
                "caching": True,
                "async_io": True,
                "performance_monitoring": True
            },
            "cache": {
                "memory_size": self.config.cache_config.memory_max_size,
                "redis_url": self.config.cache_config.redis_url,
                "compression": self.config.cache_config.enable_compression
            }
        }
        
        for key, value in common_configs.items():
            await self.config_cache.set_config(key, value)
    
    async def get_fastapi_middleware(self) -> Optional[Dict[str, Any]]:
        """Get FastAPI middleware."""
        return self.fastapi_middleware
    
    def get_database_cache(self) -> Optional[Dict[str, Any]]:
        """Get database cache integration."""
        return self.database_cache
    
    def get_session_cache(self) -> Optional[Dict[str, Any]]:
        """Get session cache integration."""
        return self.session_cache
    
    def get_config_cache(self) -> Optional[Dict[str, Any]]:
        """Get config cache integration."""
        return self.config_cache
    
    def get_template_cache(self) -> Optional[Dict[str, Any]]:
        """Get template cache integration."""
        return self.template_cache
    
    def get_asset_cache(self) -> Optional[Dict[str, Any]]:
        """Get asset cache integration."""
        return self.asset_cache
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        cache_stats = self.cache_system.get_comprehensive_stats()
        
        return {
            "cache_system": cache_stats,
            "integrations": {
                "fastapi_middleware": {
                    "cacheable_routes": len(self.fastapi_middleware.cacheable_routes)
                },
                "database_cache": {
                    "enabled": self.config.enable_database_caching
                },
                "session_cache": {
                    "enabled": self.config.enable_session_caching,
                    "session_ttl": self.config.session_ttl
                },
                "config_cache": {
                    "enabled": True
                },
                "template_cache": {
                    "enabled": True
                },
                "asset_cache": {
                    "enabled": True
                }
            }
        }

# FastAPI integration helpers
def setup_fastapi_caching(app: FastAPI, integration_manager: CacheIntegrationManager):
    """Setup FastAPI caching."""
    middleware = integration_manager.get_fastapi_middleware()
    
    # Add cacheable routes
    middleware.add_cacheable_route("/api/users", ttl=300)
    middleware.add_cacheable_route("/api/products", ttl=600)
    middleware.add_cacheable_route("/api/config", ttl=3600)
    
    # Add middleware to app
    app.middleware("http")(middleware)
    
    # Add cache endpoints
    @app.get("/api/cache/stats")
    async def get_cache_stats():
        
    """get_cache_stats function."""
return integration_manager.get_comprehensive_stats()
    
    @app.post("/api/cache/clear")
    async def clear_cache():
        
    """clear_cache function."""
await integration_manager.cache_system.clear()
        return {"message": "Cache cleared successfully"}
    
    @app.post("/api/cache/warmup")
    async def warmup_cache():
        
    """warmup_cache function."""
await integration_manager._warmup_configs()
        return {"message": "Cache warmed up successfully"}

# Dependency injection
async def get_cache_integration() -> CacheIntegrationManager:
    """Get cache integration manager dependency."""
    config = CacheIntegrationConfig()
    integration_manager = CacheIntegrationManager(config)
    await integration_manager.initialize()
    return integration_manager

# Example usage
async def example_integration_usage():
    """Example usage of cache integration."""
    
    # Create integration manager
    config = CacheIntegrationConfig(
        enable_fastapi_middleware=True,
        enable_database_caching=True,
        enable_session_caching=True
    )
    
    integration_manager = CacheIntegrationManager(config)
    await integration_manager.initialize()
    
    try:
        # Database caching
        @integration_manager.database_cache.query_cache_decorator(ttl=1800)
        async def get_user_by_id(user_id: int):
            
    """get_user_by_id function."""
# Simulate database query
            await asyncio.sleep(0.1)
            return {"id": user_id, "name": f"User {user_id}"}
        
        user_data = await get_user_by_id(123)
        logger.info(f"User data: {user_data}")
        
        # Session caching
        session_data = {"user_id": 123, "permissions": ["read", "write"]}
        await integration_manager.session_cache.set_session("session_123", session_data)
        
        retrieved_session = await integration_manager.session_cache.get_session("session_123")
        logger.info(f"Session data: {retrieved_session}")
        
        # Configuration caching
        await integration_manager.config_cache.set_config("database", {
            "host": "localhost",
            "port": 5432,
            "database": "blatam_academy"
        })
        
        db_config = await integration_manager.config_cache.get_config("database")
        logger.info(f"Database config: {db_config}")
        
        # Template caching
        async def render_email_template(template_name: str, data: dict):
            
    """render_email_template function."""
# Simulate template rendering
            await asyncio.sleep(0.1)
            return f"Hello {data.get('name', 'User')}, welcome to our platform!"
        
        email_content = await integration_manager.template_cache.render_template_cached(
            "welcome_email", {"name": "John"}, render_email_template
        )
        logger.info(f"Email content: {email_content}")
        
        # Asset caching
        async def load_image_asset(asset_path: str):
            
    """load_image_asset function."""
# Simulate asset loading
            await asyncio.sleep(0.1)
            return b"fake_image_data"
        
        image_data = await integration_manager.asset_cache.get_asset_cached(
            "/images/logo.png", load_image_asset
        )
        logger.info(f"Image data size: {len(image_data)} bytes")
        
        # Get comprehensive statistics
        stats = integration_manager.get_comprehensive_stats()
        logger.info(f"Integration statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Integration error: {e}")
    
    finally:
        await integration_manager.shutdown()

match __name__:
    case "__main__":
    asyncio.run(example_integration_usage()) 