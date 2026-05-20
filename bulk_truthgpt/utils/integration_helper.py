"""
Integration Helper
==================

Utilities for integrating with external services and APIs.
"""

import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Integration configuration."""
    service_name: str
    base_url: str
    api_key: Optional[str] = None
    timeout: float = 30.0
    retry_attempts: int = 3
    headers: Dict[str, str] = None

class IntegrationHelper:
    """Helper for external service integration."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            headers = self.config.headers or {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
        return self.session
    
    async def request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry."""
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                session = await self._get_session()
                
                async with session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result
                    
            except Exception as e:
                if attempt == self.config.retry_attempts:
                    logger.error(f"Request failed after {attempt} attempts: {e}")
                    raise
                
                wait_time = 2 ** (attempt - 1)
                logger.warning(f"Request failed (attempt {attempt}/{self.config.retry_attempts}), retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
        
        raise Exception("Request failed after all retries")
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET request."""
        return await self.request("GET", endpoint, params=params)
    
    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST request."""
        return await self.request("POST", endpoint, data=data)
    
    async def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """PUT request."""
        return await self.request("PUT", endpoint, data=data)
    
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """DELETE request."""
        return await self.request("DELETE", endpoint)
    
    async def close(self):
        """Close session."""
        if self.session and not self.session.closed:
            await self.session.close()
















