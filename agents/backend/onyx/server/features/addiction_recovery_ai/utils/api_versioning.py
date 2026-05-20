"""
API Versioning for Recovery AI
"""

from typing import Dict, List, Optional, Any, Callable
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class APIVersion:
    """API version manager"""
    
    def __init__(self):
        """Initialize API version manager"""
        self.versions: Dict[str, Dict[str, Callable]] = {}
        self.default_version = "v1"
        
        logger.info("APIVersion initialized")
    
    def register_endpoint(
        self,
        version: str,
        endpoint: str,
        handler: Callable
    ):
        """
        Register endpoint for version
        
        Args:
            version: API version (e.g., "v1", "v2")
            endpoint: Endpoint path
            handler: Handler function
        """
        if version not in self.versions:
            self.versions[version] = {}
        
        self.versions[version][endpoint] = handler
        logger.info(f"Endpoint registered: {version}/{endpoint}")
    
    def get_handler(
        self,
        version: str,
        endpoint: str
    ) -> Optional[Callable]:
        """
        Get handler for version and endpoint
        
        Args:
            version: API version
            endpoint: Endpoint path
        
        Returns:
            Handler function or None
        """
        if version in self.versions:
            return self.versions[version].get(endpoint)
        
        # Fallback to default version
        if self.default_version in self.versions:
            return self.versions[self.default_version].get(endpoint)
        
        return None
    
    def list_versions(self) -> List[str]:
        """List available versions"""
        return list(self.versions.keys())
    
    def list_endpoints(self, version: str) -> List[str]:
        """List endpoints for version"""
        if version in self.versions:
            return list(self.versions[version].keys())
        return []


def versioned(version: str):
    """
    Decorator for versioned endpoints
    
    Args:
        version: API version
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add version to kwargs
            kwargs['api_version'] = version
            return func(*args, **kwargs)
        return wrapper
    return decorator


class VersionRouter:
    """Router for API versioning"""
    
    def __init__(self, api_version: APIVersion):
        """
        Initialize version router
        
        Args:
            api_version: API version manager
        """
        self.api_version = api_version
    
    def route(
        self,
        version: str,
        endpoint: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Route request to versioned handler
        
        Args:
            version: API version
            endpoint: Endpoint path
            *args: Handler arguments
            **kwargs: Handler keyword arguments
        
        Returns:
            Handler result
        """
        handler = self.api_version.get_handler(version, endpoint)
        
        if handler:
            return handler(*args, **kwargs)
        else:
            raise ValueError(f"Handler not found: {version}/{endpoint}")

