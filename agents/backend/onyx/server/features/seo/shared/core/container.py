from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import structlog
from .config import get_settings
from .logging import get_logger
from ..http.ultra_fast_client import UltraFastHTTPClient, get_http_client, cleanup_http_client
from ..cache.ultra_fast_cache import UltraFastCache, get_cache, cleanup_cache
from ..parsers.ultra_fast_parser import UltraFastHTMLParser, get_html_parser
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Ultra-Optimized Container v10
Production-ready dependency injection with maximum performance
"""




logger = get_logger(__name__)


class Container:
    """Ultra-optimized dependency injection container"""
    
    def __init__(self) -> Any:
        self.settings = get_settings()
        self._components: Dict[str, Any] = {}
        self._initialized = False
        self._cleanup_tasks = []
    
    async def initialize(self) -> Any:
        """Initialize all components with maximum performance"""
        if self._initialized:
            return
        
        logger.info("Initializing ultra-optimized container...")
        
        try:
            # Initialize HTTP client
            http_client = await get_http_client()
            self._components["http_client"] = http_client
            logger.info("HTTP client initialized")
            
            # Initialize cache
            cache = await get_cache()
            self._components["cache"] = cache
            logger.info("Cache initialized")
            
            # Initialize HTML parser
            html_parser = get_html_parser()
            self._components["html_parser"] = html_parser
            logger.info("HTML parser initialized")
            
            # Health checks
            await self._health_check()
            
            self._initialized = True
            logger.info("Container initialized successfully")
            
        except Exception as e:
            logger.error("Container initialization failed", error=str(e))
            await self.cleanup()
            raise
    
    async def _health_check(self) -> Any:
        """Perform health checks on all components"""
        logger.info("Performing health checks...")
        
        # HTTP client health check
        http_client = self._components.get("http_client")
        if http_client:
            http_health = await http_client.health_check()
            if not http_health:
                raise RuntimeError("HTTP client health check failed")
            logger.info("HTTP client health check passed")
        
        # Cache health check
        cache = self._components.get("cache")
        if cache:
            cache_health = await cache.health_check()
            if not cache_health:
                raise RuntimeError("Cache health check failed")
            logger.info("Cache health check passed")
        
        logger.info("All health checks passed")
    
    async def cleanup(self) -> Any:
        """Cleanup all components"""
        if not self._initialized:
            return
        
        logger.info("Cleaning up container...")
        
        # Run cleanup tasks
        cleanup_tasks = []
        
        # Cleanup HTTP client
        cleanup_tasks.append(cleanup_http_client())
        
        # Cleanup cache
        cleanup_tasks.append(cleanup_cache())
        
        # Wait for all cleanup tasks
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear components
        self._components.clear()
        self._initialized = False
        
        logger.info("Container cleaned up successfully")
    
    async def get_http_client(self) -> UltraFastHTTPClient:
        """Get HTTP client instance"""
        if not self._initialized:
            raise RuntimeError("Container not initialized")
        return self._components["http_client"]
    
    def get_cache(self) -> UltraFastCache:
        """Get cache instance"""
        if not self._initialized:
            raise RuntimeError("Container not initialized")
        return self._components["cache"]
    
    def get_html_parser(self) -> UltraFastHTMLParser:
        """Get HTML parser instance"""
        if not self._initialized:
            raise RuntimeError("Container not initialized")
        return self._components["html_parser"]
    
    def get_settings(self) -> Optional[Dict[str, Any]]:
        """Get settings instance"""
        return self.settings
    
    def is_initialized(self) -> bool:
        """Check if container is initialized"""
        return self._initialized
    
    def get_component_names(self) -> list:
        """Get list of component names"""
        return list(self._components.keys())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get container metrics"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        metrics = {
            "status": "initialized",
            "component_count": len(self._components),
            "components": self.get_component_names()
        }
        
        # Add component-specific metrics
        if "http_client" in self._components:
            metrics["http_client"] = self._components["http_client"].get_metrics()
        
        if "cache" in self._components:
            metrics["cache"] = self._components["cache"].get_stats()
        
        if "html_parser" in self._components:
            metrics["html_parser"] = self._components["html_parser"].get_metrics()
        
        return metrics


# Global container instance
_global_container: Optional[Container] = None


def get_container() -> Container:
    """Get global container instance"""
    global _global_container
    if _global_container is None:
        _global_container = Container()
    return _global_container


async def initialize_container():
    """Initialize global container"""
    container = get_container()
    await container.initialize()
    return container


async def cleanup_container():
    """Cleanup global container"""
    global _global_container
    if _global_container:
        await _global_container.cleanup()
        _global_container = None


@asynccontextmanager
async def container_context():
    """Context manager for container lifecycle"""
    container = get_container()
    try:
        await container.initialize()
        yield container
    finally:
        await container.cleanup() 