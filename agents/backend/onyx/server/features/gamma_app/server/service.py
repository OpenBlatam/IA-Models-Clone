"""
Server Service Implementation
"""

from typing import Optional
import logging
from fastapi import FastAPI

from .base import ServerBase, APIRoute, Middleware, HTTPMethod

logger = logging.getLogger(__name__)


class ServerService(ServerBase):
    """Server service implementation"""
    
    def __init__(
        self,
        auth_service=None,
        access_service=None,
        chat_service=None,
        agents_service=None,
        tracing_service=None,
        config_service=None
    ):
        """Initialize server service"""
        self.auth_service = auth_service
        self.access_service = access_service
        self.chat_service = chat_service
        self.agents_service = agents_service
        self.tracing_service = tracing_service
        self.config_service = config_service
        
        self.app = FastAPI()
        self._routes: list = []
        self._middlewares: list = []
        self._running = False
    
    async def start(self) -> bool:
        """Start server"""
        try:
            # Configure FastAPI app
            self.app.title = "Gamma App API"
            self.app.version = "1.0.0"
            
            # Add middlewares
            for middleware in self._middlewares:
                try:
                    # In a real scenario, this involves mapping Middleware objects 
                    # to FastAPI BaseHTTPMiddleware or Starlette middleware classes
                    # For this implementation, we apply a placeholder application
                    logger.info(f"Adding middleware: {middleware.__class__.__name__}")
                except Exception as e:
                    logger.warning(f"Failed to add middleware: {e}")
                    
            # Add routes
            for route in self._routes:
                try:
                    # Map Custom APIRoute to FastAPI
                    logger.info(f"Adding APIRoute: {route.path} [{route.method.value}]")
                    self.app.add_api_route(
                        path=route.path,
                        endpoint=route.endpoint,
                        methods=[route.method.value],
                        tags=route.tags
                    )
                except Exception as e:
                    logger.warning(f"Failed to add route {route.path}: {e}")
            
            self._running = True
            logger.info("FastAPI Server configured and started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop server"""
        try:
            self._running = False
            logger.info("FastAPI Server stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            return False
    
    def add_route(self, route: APIRoute) -> bool:
        """Add route"""
        try:
            self._routes.append(route)
            
            # If server is already running, apply dynamic route
            if self._running:
                self.app.add_api_route(
                    path=route.path,
                    endpoint=route.endpoint,
                    methods=[route.method.value],
                    tags=route.tags
                )
            return True
            
        except Exception as e:
            logger.error(f"Error adding route: {e}")
            return False
    
    def add_middleware(self, middleware: Middleware) -> bool:
        """Add middleware"""
        try:
            self._middlewares.append(middleware)
            return True
            
        except Exception as e:
            logger.error(f"Error adding middleware: {e}")
            return False

