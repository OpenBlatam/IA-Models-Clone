"""
Endpoint Builder
================

Builder pattern for creating API endpoints.
"""

from typing import Dict, Any, Optional, List, Callable
from fastapi import Depends
from pydantic import BaseModel
from aws.modules.presentation.api_router import APIRouter


class EndpointBuilder:
    """Builder for API endpoints."""
    
    def __init__(self, router: APIRouter):
        self.router = router
        self._path: Optional[str] = None
        self._method: Optional[str] = None
        self._handler: Optional[Callable] = None
        self._dependencies: List[Depends] = []
        self._response_model: Optional[BaseModel] = None
        self._summary: Optional[str] = None
        self._description: Optional[str] = None
    
    def path(self, path: str) -> "EndpointBuilder":
        """Set endpoint path."""
        self._path = path
        return self
    
    def method(self, method: str) -> "EndpointBuilder":
        """Set HTTP method."""
        self._method = method.upper()
        return self
    
    def handler(self, handler: Callable) -> "EndpointBuilder":
        """Set handler function."""
        self._handler = handler
        return self
    
    def dependency(self, dependency: Depends) -> "EndpointBuilder":
        """Add dependency."""
        self._dependencies.append(dependency)
        return self
    
    def response_model(self, model: BaseModel) -> "EndpointBuilder":
        """Set response model."""
        self._response_model = model
        return self
    
    def summary(self, summary: str) -> "EndpointBuilder":
        """Set endpoint summary."""
        self._summary = summary
        return self
    
    def description(self, description: str) -> "EndpointBuilder":
        """Set endpoint description."""
        self._description = description
        return self
    
    def build(self):
        """Build and register endpoint."""
        if not all([self._path, self._method, self._handler]):
            raise ValueError("Path, method, and handler are required")
        
        self.router.register_endpoint(
            path=self._path,
            method=self._method,
            handler=self._handler,
            dependencies=self._dependencies if self._dependencies else None,
            response_model=self._response_model,
            summary=self._summary,
            description=self._description
        )
        
        # Reset builder
        self._reset()
    
    def _reset(self):
        """Reset builder state."""
        self._path = None
        self._method = None
        self._handler = None
        self._dependencies = []
        self._response_model = None
        self._summary = None
        self._description = None















