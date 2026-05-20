"""
HTTPX Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class HTTPMethod(str, Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class HTTPRequest:
    """HTTP request"""
    method: HTTPMethod
    url: str
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    json: Optional[Dict[str, Any]] = None
    data: Optional[bytes] = None
    timeout: int = 30


@dataclass
class HTTPResponse:
    """HTTP response"""
    status_code: int
    headers: Dict[str, str]
    content: bytes
    json: Optional[Dict[str, Any]] = None


class HTTPClient:
    """HTTP client definition"""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ):
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.timeout = timeout


class HTTPClientBase(ABC):
    """Base interface for HTTP client"""
    
    @abstractmethod
    async def request(self, request: HTTPRequest) -> HTTPResponse:
        """Make HTTP request"""
        pass
    
    @abstractmethod
    async def get(self, url: str, **kwargs) -> HTTPResponse:
        """GET request"""
        pass
    
    @abstractmethod
    async def post(self, url: str, **kwargs) -> HTTPResponse:
        """POST request"""
        pass

