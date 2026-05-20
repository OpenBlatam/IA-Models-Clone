"""
API Versioning System
=====================

Advanced system for API versioning and backward compatibility.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class VersionStrategy(Enum):
    """Version strategy."""
    URL_PATH = "url_path"  # /v1/endpoint
    QUERY_PARAM = "query_param"  # /endpoint?version=1
    HEADER = "header"  # X-API-Version: 1
    ACCEPT_HEADER = "accept_header"  # Accept: application/vnd.api+json;version=1


@dataclass
class APIVersion:
    """API version definition."""
    version: str
    deprecated: bool = False
    deprecated_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    changelog: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)


@dataclass
class VersionedEndpoint:
    """Versioned endpoint definition."""
    path: str
    method: str
    handler: Callable
    versions: Dict[str, APIVersion]
    default_version: str = "1.0"


class APIVersionManager:
    """API version manager."""
    
    def __init__(self, default_version: str = "1.0", strategy: VersionStrategy = VersionStrategy.URL_PATH):
        """
        Initialize API version manager.
        
        Args:
            default_version: Default API version
            strategy: Version detection strategy
        """
        self.default_version = default_version
        self.strategy = strategy
        self.versions: Dict[str, APIVersion] = {}
        self.endpoints: Dict[str, VersionedEndpoint] = {}
        self.middleware: List[Callable] = []
    
    def register_version(self, version: str, api_version: APIVersion):
        """
        Register an API version.
        
        Args:
            version: Version string
            api_version: API version definition
        """
        self.versions[version] = api_version
        logger.info(f"Registered API version: {version}")
    
    def register_endpoint(
        self,
        path: str,
        method: str,
        handler: Callable,
        versions: Dict[str, APIVersion],
        default_version: Optional[str] = None
    ):
        """
        Register a versioned endpoint.
        
        Args:
            path: Endpoint path
            method: HTTP method
            handler: Handler function
            versions: Dictionary of version -> APIVersion
            default_version: Default version for this endpoint
        """
        endpoint_key = f"{method.upper()}:{path}"
        self.endpoints[endpoint_key] = VersionedEndpoint(
            path=path,
            method=method,
            handler=handler,
            versions=versions,
            default_version=default_version or self.default_version
        )
        logger.info(f"Registered versioned endpoint: {endpoint_key}")
    
    def get_version_from_request(self, request: Any) -> str:
        """
        Extract version from request.
        
        Args:
            request: Request object
            
        Returns:
            Version string
        """
        if self.strategy == VersionStrategy.URL_PATH:
            # Extract from URL path /v1/endpoint
            path = getattr(request, 'url', {}).path if hasattr(request, 'url') else str(request.url.path)
            parts = path.split('/')
            for part in parts:
                if part.startswith('v') and part[1:].replace('.', '').isdigit():
                    return part[1:]
        
        elif self.strategy == VersionStrategy.QUERY_PARAM:
            # Extract from query parameter
            query_params = getattr(request, 'query_params', {})
            if hasattr(query_params, 'get'):
                version = query_params.get('version')
                if version:
                    return version
        
        elif self.strategy == VersionStrategy.HEADER:
            # Extract from header
            headers = getattr(request, 'headers', {})
            version = headers.get('X-API-Version') or headers.get('x-api-version')
            if version:
                return version
        
        elif self.strategy == VersionStrategy.ACCEPT_HEADER:
            # Extract from Accept header
            headers = getattr(request, 'headers', {})
            accept = headers.get('Accept') or headers.get('accept', '')
            # Parse Accept: application/vnd.api+json;version=1
            if 'version=' in accept:
                version_part = accept.split('version=')[1].split(';')[0].split(',')[0]
                return version_part.strip()
        
        return self.default_version
    
    def get_endpoint_handler(self, path: str, method: str, version: Optional[str] = None) -> Optional[Callable]:
        """
        Get handler for versioned endpoint.
        
        Args:
            path: Endpoint path
            method: HTTP method
            version: Optional version (uses default if not provided)
            
        Returns:
            Handler function or None
        """
        endpoint_key = f"{method.upper()}:{path}"
        if endpoint_key not in self.endpoints:
            return None
        
        endpoint = self.endpoints[endpoint_key]
        version = version or endpoint.default_version
        
        if version in endpoint.versions:
            return endpoint.handler
        
        # Fallback to default version
        if endpoint.default_version in endpoint.versions:
            return endpoint.handler
        
        return None
    
    def check_deprecation(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Check if version is deprecated.
        
        Args:
            version: Version string
            
        Returns:
            Deprecation info or None
        """
        if version not in self.versions:
            return None
        
        api_version = self.versions[version]
        if not api_version.deprecated:
            return None
        
        info = {
            "deprecated": True,
            "version": version,
            "deprecated_date": api_version.deprecated_date.isoformat() if api_version.deprecated_date else None,
            "sunset_date": api_version.sunset_date.isoformat() if api_version.sunset_date else None,
            "migration_guide": api_version.changelog
        }
        
        # Check if sunset date has passed
        if api_version.sunset_date and datetime.now() > api_version.sunset_date:
            info["sunset"] = True
        
        return info
    
    def version_decorator(self, versions: List[str], default_version: Optional[str] = None):
        """
        Decorator for versioning endpoints.
        
        Args:
            versions: List of supported versions
            default_version: Default version
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract version from request
                request = kwargs.get('request') or (args[0] if args else None)
                version = self.get_version_from_request(request) if request else default_version or self.default_version
                
                # Check deprecation
                deprecation_info = self.check_deprecation(version)
                if deprecation_info:
                    logger.warning(f"Using deprecated version: {version}")
                
                # Add version to kwargs
                kwargs['api_version'] = version
                if deprecation_info:
                    kwargs['deprecation_info'] = deprecation_info
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_version_info(self) -> Dict[str, Any]:
        """
        Get version information.
        
        Returns:
            Version information dictionary
        """
        return {
            "default_version": self.default_version,
            "strategy": self.strategy.value,
            "versions": {
                version: {
                    "deprecated": api_version.deprecated,
                    "deprecated_date": api_version.deprecated_date.isoformat() if api_version.deprecated_date else None,
                    "sunset_date": api_version.sunset_date.isoformat() if api_version.sunset_date else None,
                    "changelog": api_version.changelog,
                    "breaking_changes": api_version.breaking_changes
                }
                for version, api_version in self.versions.items()
            },
            "endpoints": len(self.endpoints)
        }



