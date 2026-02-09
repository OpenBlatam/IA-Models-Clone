"""
Request Builder Utilities for Piel Mejorador AI SAM3
====================================================

Unified HTTP request building utilities.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


@dataclass
class HTTPRequest:
    """HTTP request definition."""
    method: str = "GET"
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    json: Optional[Dict[str, Any]] = None
    data: Optional[Union[str, bytes, Dict[str, Any]]] = None
    files: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    auth: Optional[tuple] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for httpx."""
        result = {
            "method": self.method,
            "url": self.url,
            "headers": self.headers,
        }
        
        if self.params:
            result["params"] = self.params
        
        if self.json:
            result["json"] = self.json
        
        if self.data:
            result["data"] = self.data
        
        if self.files:
            result["files"] = self.files
        
        if self.timeout:
            result["timeout"] = self.timeout
        
        if self.auth:
            result["auth"] = self.auth
        
        return result


class RequestBuilderUtils:
    """Unified request building utilities."""
    
    @staticmethod
    def create_request(
        method: str = "GET",
        url: str = "",
        **kwargs
    ) -> HTTPRequest:
        """
        Create HTTP request.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            HTTPRequest
        """
        return HTTPRequest(method=method, url=url, **kwargs)
    
    @staticmethod
    def build_get_request(
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> HTTPRequest:
        """
        Build GET request.
        
        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            
        Returns:
            HTTPRequest
        """
        return HTTPRequest(
            method="GET",
            url=url,
            params=params or {},
            headers=headers or {}
        )
    
    @staticmethod
    def build_post_request(
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> HTTPRequest:
        """
        Build POST request.
        
        Args:
            url: Request URL
            json: JSON body
            data: Form data or raw data
            headers: Request headers
            params: Query parameters
            
        Returns:
            HTTPRequest
        """
        return HTTPRequest(
            method="POST",
            url=url,
            json=json,
            data=data,
            headers=headers or {},
            params=params or {}
        )
    
    @staticmethod
    def build_put_request(
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> HTTPRequest:
        """
        Build PUT request.
        
        Args:
            url: Request URL
            json: JSON body
            data: Form data or raw data
            headers: Request headers
            
        Returns:
            HTTPRequest
        """
        return HTTPRequest(
            method="PUT",
            url=url,
            json=json,
            data=data,
            headers=headers or {}
        )
    
    @staticmethod
    def build_delete_request(
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> HTTPRequest:
        """
        Build DELETE request.
        
        Args:
            url: Request URL
            headers: Request headers
            params: Query parameters
            
        Returns:
            HTTPRequest
        """
        return HTTPRequest(
            method="DELETE",
            url=url,
            headers=headers or {},
            params=params or {}
        )
    
    @staticmethod
    def add_auth(
        request: HTTPRequest,
        auth_type: str = "bearer",
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> HTTPRequest:
        """
        Add authentication to request.
        
        Args:
            request: HTTP request
            auth_type: "bearer", "basic", or "custom"
            token: Bearer token
            username: Basic auth username
            password: Basic auth password
            
        Returns:
            Request with auth
        """
        if auth_type == "bearer" and token:
            request.headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "basic" and username and password:
            import base64
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            request.headers["Authorization"] = f"Basic {credentials}"
            request.auth = (username, password)
        
        return request
    
    @staticmethod
    def add_headers(
        request: HTTPRequest,
        headers: Dict[str, str]
    ) -> HTTPRequest:
        """
        Add headers to request.
        
        Args:
            request: HTTP request
            headers: Headers to add
            
        Returns:
            Request with headers
        """
        request.headers.update(headers)
        return request
    
    @staticmethod
    def add_query_params(
        request: HTTPRequest,
        params: Dict[str, Any]
    ) -> HTTPRequest:
        """
        Add query parameters to request.
        
        Args:
            request: HTTP request
            params: Query parameters
            
        Returns:
            Request with query params
        """
        request.params.update(params)
        return request
    
    @staticmethod
    def set_json_body(
        request: HTTPRequest,
        data: Dict[str, Any]
    ) -> HTTPRequest:
        """
        Set JSON body for request.
        
        Args:
            request: HTTP request
            data: JSON data
            
        Returns:
            Request with JSON body
        """
        request.json = data
        request.headers["Content-Type"] = "application/json"
        return request
    
    @staticmethod
    def set_form_data(
        request: HTTPRequest,
        data: Dict[str, Any]
    ) -> HTTPRequest:
        """
        Set form data for request.
        
        Args:
            request: HTTP request
            data: Form data
            
        Returns:
            Request with form data
        """
        request.data = data
        request.headers["Content-Type"] = "application/x-www-form-urlencoded"
        return request
    
    @staticmethod
    def set_timeout(
        request: HTTPRequest,
        timeout: float
    ) -> HTTPRequest:
        """
        Set timeout for request.
        
        Args:
            request: HTTP request
            timeout: Timeout in seconds
            
        Returns:
            Request with timeout
        """
        request.timeout = timeout
        return request


class RequestBuilder:
    """Fluent builder for HTTP requests."""
    
    def __init__(self, method: str = "GET", url: str = ""):
        """
        Initialize request builder.
        
        Args:
            method: HTTP method
            url: Request URL
        """
        self._request = HTTPRequest(method=method, url=url)
    
    def url(self, url: str) -> "RequestBuilder":
        """Set request URL."""
        self._request.url = url
        return self
    
    def method(self, method: str) -> "RequestBuilder":
        """Set HTTP method."""
        self._request.method = method
        return self
    
    def header(self, key: str, value: str) -> "RequestBuilder":
        """Add header."""
        self._request.headers[key] = value
        return self
    
    def headers(self, headers: Dict[str, str]) -> "RequestBuilder":
        """Add headers."""
        self._request.headers.update(headers)
        return self
    
    def param(self, key: str, value: Any) -> "RequestBuilder":
        """Add query parameter."""
        self._request.params[key] = value
        return self
    
    def params(self, params: Dict[str, Any]) -> "RequestBuilder":
        """Add query parameters."""
        self._request.params.update(params)
        return self
    
    def json(self, data: Dict[str, Any]) -> "RequestBuilder":
        """Set JSON body."""
        self._request.json = data
        self._request.headers["Content-Type"] = "application/json"
        return self
    
    def data(self, data: Union[str, bytes, Dict[str, Any]]) -> "RequestBuilder":
        """Set data body."""
        self._request.data = data
        return self
    
    def auth_bearer(self, token: str) -> "RequestBuilder":
        """Add bearer token authentication."""
        self._request.headers["Authorization"] = f"Bearer {token}"
        return self
    
    def auth_basic(self, username: str, password: str) -> "RequestBuilder":
        """Add basic authentication."""
        import base64
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        self._request.headers["Authorization"] = f"Basic {credentials}"
        self._request.auth = (username, password)
        return self
    
    def timeout(self, seconds: float) -> "RequestBuilder":
        """Set timeout."""
        self._request.timeout = seconds
        return self
    
    def build(self) -> HTTPRequest:
        """
        Build HTTP request.
        
        Returns:
            HTTPRequest
        """
        return self._request


# Convenience functions
def create_request(method: str = "GET", url: str = "", **kwargs) -> HTTPRequest:
    """Create HTTP request."""
    return RequestBuilderUtils.create_request(method, url, **kwargs)


def build_get_request(url: str, **kwargs) -> HTTPRequest:
    """Build GET request."""
    return RequestBuilderUtils.build_get_request(url, **kwargs)


def build_post_request(url: str, **kwargs) -> HTTPRequest:
    """Build POST request."""
    return RequestBuilderUtils.build_post_request(url, **kwargs)


def request_builder(method: str = "GET", url: str = "") -> RequestBuilder:
    """Create request builder."""
    return RequestBuilder(method, url)




