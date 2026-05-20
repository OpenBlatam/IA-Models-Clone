"""
Integration Helper for Document Analyzer
=========================================

Advanced integration utilities for external services.
"""

import asyncio
import logging
import aiohttp
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Integration configuration"""
    base_url: str
    timeout: float = 30.0
    retry_count: int = 3
    retry_backoff: float = 1.0
    headers: Dict[str, str] = None

class IntegrationHelper:
    """Advanced integration helper"""
    
    def __init__(self):
        self.configs: Dict[str, IntegrationConfig] = {}
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        logger.info("IntegrationHelper initialized")
    
    def register_integration(
        self,
        name: str,
        config: IntegrationConfig
    ):
        """Register an integration"""
        self.configs[name] = config
        logger.info(f"Registered integration: {name}")
    
    async def get_session(self, name: str) -> aiohttp.ClientSession:
        """Get or create HTTP session for integration"""
        if name not in self.sessions:
            config = self.configs.get(name)
            timeout = aiohttp.ClientTimeout(total=config.timeout if config else 30.0)
            self.sessions[name] = aiohttp.ClientSession(timeout=timeout)
        return self.sessions[name]
    
    async def request(
        self,
        integration_name: str,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request to integration"""
        if integration_name not in self.configs:
            raise ValueError(f"Integration {integration_name} not registered")
        
        config = self.configs[integration_name]
        session = await self.get_session(integration_name)
        
        url = f"{config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        request_headers = {**(config.headers or {}), **(headers or {})}
        
        for attempt in range(config.retry_count):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=request_headers,
                    **kwargs
                ) as response:
                    result = {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "data": await response.json() if response.content_type == 'application/json' else await response.text()
                    }
                    
                    response.raise_for_status()
                    return result
            
            except Exception as e:
                if attempt < config.retry_count - 1:
                    wait_time = config.retry_backoff * (2 ** attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}/{config.retry_count}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {config.retry_count} attempts: {e}")
                    raise
    
    async def close_sessions(self):
        """Close all HTTP sessions"""
        for name, session in self.sessions.items():
            await session.close()
        self.sessions.clear()
        logger.info("Closed all integration sessions")

# Global instance
integration_helper = IntegrationHelper()
















