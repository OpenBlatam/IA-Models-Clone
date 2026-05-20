"""
API Module
Independent API module for FastAPI routes
"""

from typing import List
from fastapi import APIRouter
from modules.base_module import BaseModule

logger = __import__("logging").getLogger(__name__)


class APIModule(BaseModule):
    """API feature module"""
    
    def __init__(self):
        super().__init__("api", "1.0.0")
        self._router = APIRouter()
        self._routes_registered = False
    
    def get_dependencies(self) -> List[str]:
        """API module depends on storage, cache, security"""
        return ["storage", "cache", "security"]
    
    def _on_initialize(self) -> None:
        """Initialize API module"""
        if not self._routes_registered:
            self._register_routes()
            self._routes_registered = True
        
        logger.info("API module initialized")
    
    def _on_shutdown(self) -> None:
        """Shutdown API module"""
        logger.info("API module shut down")
    
    def _register_routes(self) -> None:
        """Register API routes (override in subclasses)"""
        # Base implementation - override in feature-specific modules
        pass
    
    def get_router(self) -> APIRouter:
        """Get API router"""
        return self._router
    
    def include_router(self, router: APIRouter, **kwargs) -> None:
        """Include another router"""
        self._router.include_router(router, **kwargs)















