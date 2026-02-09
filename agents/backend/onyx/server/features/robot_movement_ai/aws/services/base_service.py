"""
Base Service
============

Base class for all microservices following stateless principles.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ServiceConfig(BaseModel):
    """Service configuration."""
    service_name: str
    service_version: str = "1.0.0"
    port: int = 8000
    host: str = "0.0.0.0"
    health_check_path: str = "/health"
    metrics_path: str = "/metrics"
    enable_cors: bool = True
    cors_origins: List[str] = ["*"]


class BaseMicroservice(ABC):
    """Base class for all microservices."""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.app: Optional[FastAPI] = None
        self._initialized = False
    
    @abstractmethod
    def create_app(self) -> FastAPI:
        """Create FastAPI application for this service."""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> Dict[str, Any]:
        """Get service dependencies (for dependency injection)."""
        pass
    
    def setup(self) -> FastAPI:
        """Setup the service."""
        if self._initialized:
            return self.app
        
        self.app = self.create_app()
        self._setup_routes()
        self._setup_middleware()
        self._setup_health_check()
        self._initialized = True
        
        logger.info(f"Service {self.config.service_name} initialized")
        return self.app
    
    def _setup_routes(self):
        """Setup service routes."""
        # Override in subclasses
        pass
    
    def _setup_middleware(self):
        """Setup service middleware."""
        from fastapi.middleware.cors import CORSMiddleware
        
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    def _setup_health_check(self):
        """Setup health check endpoint."""
        @self.app.get(self.config.health_check_path)
        async def health_check():
            return {
                "status": "healthy",
                "service": self.config.service_name,
                "version": self.config.service_version
            }
    
    def get_app(self) -> FastAPI:
        """Get FastAPI app."""
        if not self._initialized:
            return self.setup()
        return self.app
    
    def is_stateless(self) -> bool:
        """Check if service is stateless."""
        return True  # All services should be stateless
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            "name": self.config.service_name,
            "version": self.config.service_version,
            "stateless": self.is_stateless(),
            "endpoints": [
                route.path for route in self.app.routes if hasattr(route, "path")
            ] if self.app else []
        }















