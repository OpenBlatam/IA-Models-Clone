"""
Server Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any
from enum import Enum


class HTTPMethod(str, Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class APIRoute:
    """API route definition"""
    
    def __init__(
        self,
        path: str,
        method: HTTPMethod,
        handler: Callable,
        dependencies: Optional[list] = None
    ):
        self.path = path
        self.method = method
        self.handler = handler
        self.dependencies = dependencies or []


class Middleware:
    """Middleware definition"""
    
    def __init__(
        self,
        name: str,
        handler: Callable,
        order: int = 0
    ):
        self.name = name
        self.handler = handler
        self.order = order


class ServerBase(ABC):
    """Base interface for server"""
    
    @abstractmethod
    async def start(self) -> bool:
        """Start server"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop server"""
        pass
    
    @abstractmethod
    def add_route(self, route: APIRoute) -> bool:
        """Add route"""
        pass
    
    @abstractmethod
    def add_middleware(self, middleware: Middleware) -> bool:
        """Add middleware"""
        pass

