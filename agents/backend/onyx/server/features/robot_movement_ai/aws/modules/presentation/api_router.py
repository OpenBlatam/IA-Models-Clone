"""
API Router
==========

Modular API router for FastAPI endpoints.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from fastapi import APIRouter, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class APIRouter:
    """Modular API router."""
    
    def __init__(self, prefix: str = "", tags: Optional[List[str]] = None):
        self.prefix = prefix
        self.tags = tags or []
        self.router = APIRouter(prefix=prefix, tags=tags)
        self._endpoints: Dict[str, Dict[str, Any]] = {}
    
    def register_endpoint(
        self,
        path: str,
        method: str,
        handler: Callable,
        dependencies: Optional[List[Depends]] = None,
        response_model: Optional[BaseModel] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None
    ):
        """Register an endpoint."""
        endpoint_key = f"{method.upper()}:{path}"
        
        # Add to router
        if method.upper() == "GET":
            self.router.get(
                path,
                dependencies=dependencies,
                response_model=response_model,
                summary=summary,
                description=description
            )(handler)
        elif method.upper() == "POST":
            self.router.post(
                path,
                dependencies=dependencies,
                response_model=response_model,
                summary=summary,
                description=description
            )(handler)
        elif method.upper() == "PUT":
            self.router.put(
                path,
                dependencies=dependencies,
                response_model=response_model,
                summary=summary,
                description=description
            )(handler)
        elif method.upper() == "DELETE":
            self.router.delete(
                path,
                dependencies=dependencies,
                response_model=response_model,
                summary=summary,
                description=description
            )(handler)
        
        self._endpoints[endpoint_key] = {
            "path": path,
            "method": method,
            "handler": handler,
            "summary": summary
        }
        
        logger.debug(f"Registered endpoint: {method} {self.prefix}{path}")
    
    def get_router(self) -> APIRouter:
        """Get FastAPI router."""
        return self.router
    
    def get_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered endpoints."""
        return self._endpoints.copy()















