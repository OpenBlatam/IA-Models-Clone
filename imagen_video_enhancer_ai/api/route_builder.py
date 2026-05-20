"""
Route Builder
=============

Builder pattern for creating API routes.
"""

from typing import Callable, Optional, List, Dict, Any
from fastapi import APIRouter, Depends
from fastapi.routing import APIRoute


class RouteBuilder:
    """Builder for API routes."""
    
    def __init__(self, router: APIRouter):
        """
        Initialize route builder.
        
        Args:
            router: FastAPI router
        """
        self.router = router
        self._path: Optional[str] = None
        self._method: Optional[str] = None
        self._handler: Optional[Callable] = None
        self._dependencies: List[Depends] = []
        self._tags: List[str] = []
        self._summary: Optional[str] = None
        self._description: Optional[str] = None
        self._response_model: Optional[type] = None
        self._status_code: int = 200
        self._name: Optional[str] = None
    
    def path(self, path: str) -> "RouteBuilder":
        """
        Set route path.
        
        Args:
            path: Route path
            
        Returns:
            Self for chaining
        """
        self._path = path
        return self
    
    def method(self, method: str) -> "RouteBuilder":
        """
        Set HTTP method.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            
        Returns:
            Self for chaining
        """
        self._method = method.upper()
        return self
    
    def handler(self, handler: Callable) -> "RouteBuilder":
        """
        Set route handler.
        
        Args:
            handler: Route handler function
            
        Returns:
            Self for chaining
        """
        self._handler = handler
        return self
    
    def depends(self, *dependencies: Depends) -> "RouteBuilder":
        """
        Add dependencies.
        
        Args:
            *dependencies: Dependency objects
            
        Returns:
            Self for chaining
        """
        self._dependencies.extend(dependencies)
        return self
    
    def tags(self, *tags: str) -> "RouteBuilder":
        """
        Add tags.
        
        Args:
            *tags: Tag names
            
        Returns:
            Self for chaining
        """
        self._tags.extend(tags)
        return self
    
    def summary(self, summary: str) -> "RouteBuilder":
        """
        Set route summary.
        
        Args:
            summary: Route summary
            
        Returns:
            Self for chaining
        """
        self._summary = summary
        return self
    
    def description(self, description: str) -> "RouteBuilder":
        """
        Set route description.
        
        Args:
            description: Route description
            
        Returns:
            Self for chaining
        """
        self._description = description
        return self
    
    def response_model(self, model: type) -> "RouteBuilder":
        """
        Set response model.
        
        Args:
            model: Response model class
            
        Returns:
            Self for chaining
        """
        self._response_model = model
        return self
    
    def status_code(self, code: int) -> "RouteBuilder":
        """
        Set status code.
        
        Args:
            code: HTTP status code
            
        Returns:
            Self for chaining
        """
        self._status_code = code
        return self
    
    def name(self, name: str) -> "RouteBuilder":
        """
        Set route name.
        
        Args:
            name: Route name
            
        Returns:
            Self for chaining
        """
        self._name = name
        return self
    
    def build(self) -> APIRoute:
        """
        Build and register route.
        
        Returns:
            Created route
            
        Raises:
            ValueError: If required fields are missing
        """
        if not self._path:
            raise ValueError("Path is required")
        if not self._method:
            raise ValueError("Method is required")
        if not self._handler:
            raise ValueError("Handler is required")
        
        # Get router method
        router_method = getattr(self.router, self._method.lower())
        
        # Build route arguments
        kwargs: Dict[str, Any] = {
            "path": self._path,
            "dependencies": self._dependencies if self._dependencies else None,
            "tags": self._tags if self._tags else None,
            "summary": self._summary,
            "description": self._description,
            "response_model": self._response_model,
            "status_code": self._status_code,
            "name": self._name
        }
        
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Register route
        route = router_method(self._handler, **kwargs)
        
        # Reset builder
        self._reset()
        
        return route
    
    def _reset(self):
        """Reset builder state."""
        self._path = None
        self._method = None
        self._handler = None
        self._dependencies = []
        self._tags = []
        self._summary = None
        self._description = None
        self._response_model = None
        self._status_code = 200
        self._name = None




