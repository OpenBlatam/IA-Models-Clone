"""REST API client for external integrations"""
from typing import Dict, Any, Optional, List
import httpx
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class APIClient:
    """REST API client for external integrations"""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize API client
        
        Args:
            base_url: Base URL for API
            api_key: Optional API key
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers()
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get default headers"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MarkdownToDocsAI/1.9.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        GET request
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data
        """
        try:
            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"API GET error: {e}")
            raise
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        POST request
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            
        Returns:
            Response data
        """
        try:
            if json_data:
                response = await self.client.post(endpoint, json=json_data)
            else:
                response = await self.client.post(endpoint, data=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"API POST error: {e}")
            raise
    
    async def put(
        self,
        endpoint: str,
        json_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        PUT request
        
        Args:
            endpoint: API endpoint
            json_data: JSON data
            
        Returns:
            Response data
        """
        try:
            response = await self.client.put(endpoint, json=json_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"API PUT error: {e}")
            raise
    
    async def delete(
        self,
        endpoint: str
    ) -> Dict[str, Any]:
        """
        DELETE request
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Response data
        """
        try:
            response = await self.client.delete(endpoint)
            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPError as e:
            logger.error(f"API DELETE error: {e}")
            raise
    
    async def close(self):
        """Close client"""
        await self.client.aclose()


class IntegrationManager:
    """Manage external API integrations"""
    
    def __init__(self):
        self.clients: Dict[str, APIClient] = {}
        self.integrations: Dict[str, Dict[str, Any]] = {}
    
    def register_integration(
        self,
        name: str,
        base_url: str,
        api_key: Optional[str] = None
    ) -> bool:
        """
        Register an integration
        
        Args:
            name: Integration name
            base_url: Base URL
            api_key: Optional API key
            
        Returns:
            True if successful
        """
        try:
            client = APIClient(base_url, api_key)
            self.clients[name] = client
            self.integrations[name] = {
                "base_url": base_url,
                "registered_at": datetime.now().isoformat()
            }
            return True
        except Exception as e:
            logger.error(f"Error registering integration: {e}")
            return False
    
    def get_client(self, name: str) -> Optional[APIClient]:
        """Get API client for integration"""
        return self.clients.get(name)
    
    def list_integrations(self) -> List[str]:
        """List registered integrations"""
        return list(self.integrations.keys())
    
    async def close_all(self):
        """Close all clients"""
        for client in self.clients.values():
            await client.close()
        self.clients.clear()


# Global integration manager
_integration_manager: Optional[IntegrationManager] = None


def get_integration_manager() -> IntegrationManager:
    """Get global integration manager"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = IntegrationManager()
    return _integration_manager

