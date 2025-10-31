from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any, Optional
from loguru import logger
from ..core.ultra_optimized_http_client import UltraOptimizedHTTPClient as CoreHTTPClient
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
HTTP Client adapter for infrastructure layer.
Wraps the core HTTP client with infrastructure-specific logic.
"""




class UltraOptimizedHTTPClient:
    """HTTP Client adapter for infrastructure layer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.core_client = CoreHTTPClient(config)
        self._setup_infrastructure()
    
    def _setup_infrastructure(self) -> Any:
        """Setup infrastructure-specific configurations."""
        # Configurar headers específicos de infraestructura
        infrastructure_headers = {
            'X-Infrastructure': 'seo-service',
            'X-Version': '2.0.0',
            'X-Environment': self.config.get('environment', 'production')
        }
        
        # Agregar headers de infraestructura
        self.core_client.default_headers.update(infrastructure_headers)
        
        logger.info("Infrastructure HTTP client configured")
    
    async def get(self, url: str, headers: Optional[Dict[str, str]] = None, 
                  timeout: Optional[float] = None):
        """GET request with infrastructure logging."""
        logger.debug(f"Infrastructure GET request: {url}")
        return await self.core_client.get(url, headers, timeout)
    
    async def post(self, url: str, data: Any = None, json: Any = None,
                   headers: Optional[Dict[str, str]] = None,
                   timeout: Optional[float] = None):
        """POST request with infrastructure logging."""
        logger.debug(f"Infrastructure POST request: {url}")
        return await self.core_client.post(url, data, json, headers, timeout)
    
    async def get_many(self, urls: list, headers: Optional[Dict[str, str]] = None,
                       max_concurrent: Optional[int] = None):
        """Multiple GET requests with infrastructure logging."""
        logger.debug(f"Infrastructure batch GET request: {len(urls)} URLs")
        return await self.core_client.get_many(urls, headers, max_concurrent)
    
    async def head(self, url: str, headers: Optional[Dict[str, str]] = None,
                   timeout: Optional[float] = None):
        """HEAD request with infrastructure logging."""
        logger.debug(f"Infrastructure HEAD request: {url}")
        return await self.core_client.head(url, headers, timeout)
    
    async def check_url_health(self, url: str, timeout: float = 5.0):
        """Health check with infrastructure logging."""
        logger.debug(f"Infrastructure health check: {url}")
        return await self.core_client.check_url_health(url, timeout)
    
    async def download_file(self, url: str, filepath: str, chunk_size: int = 8192):
        """File download with infrastructure logging."""
        logger.debug(f"Infrastructure file download: {url} -> {filepath}")
        return await self.core_client.download_file(url, filepath, chunk_size)
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get stats with infrastructure context."""
        stats = self.core_client.get_stats()
        stats['infrastructure_layer'] = True
        return stats
    
    async def health_check(self) -> Any:
        """Health check with infrastructure context."""
        health = await self.core_client.health_check()
        health['infrastructure'] = 'ok'
        return health
    
    async def close(self) -> Any:
        """Close with infrastructure cleanup."""
        logger.info("Closing infrastructure HTTP client")
        await self.core_client.close()
    
    async def __aenter__(self) -> Any:
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        await self.close() 