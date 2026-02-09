"""
API Gateway
===========

API Gateway service that routes requests to microservices.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from aws.services.base_service import BaseMicroservice, ServiceConfig
from aws.services.service_client import ServiceClientFactory, ServiceClient
from aws.services.service_registry import get_service_registry
from aws.core.config_manager import AppConfig

logger = logging.getLogger(__name__)


class APIGatewayService(BaseMicroservice):
    """API Gateway microservice."""
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        if config is None:
            config = ServiceConfig(
                service_name="api-gateway",
                service_version="1.0.0",
                port=8000
            )
        super().__init__(config)
        self.registry = get_service_registry()
        self._routes: Dict[str, str] = {
            "/api/v1/move": "movement-service",
            "/api/v1/chat": "chat-service",
            "/api/v1/trajectory": "trajectory-service",
        }
    
    def create_app(self) -> FastAPI:
        """Create FastAPI app for API Gateway."""
        app = FastAPI(
            title="Robot Movement AI API Gateway",
            description="API Gateway for microservices",
            version=self.config.service_version
        )
        return app
    
    def get_dependencies(self) -> Dict[str, Any]:
        """Get service dependencies."""
        return {}
    
    def _setup_routes(self):
        """Setup API Gateway routes."""
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
        async def gateway_route(request: Request, path: str):
            """Route requests to appropriate microservice."""
            # Find target service
            target_service = self._find_target_service(path)
            
            if not target_service:
                raise HTTPException(status_code=404, detail="Service not found")
            
            # Get service client
            client = ServiceClientFactory.get_client(target_service)
            
            try:
                # Forward request
                method = request.method
                body = await request.body() if method in ["POST", "PUT", "PATCH"] else None
                params = dict(request.query_params)
                headers = dict(request.headers)
                
                # Remove host header
                headers.pop("host", None)
                headers.pop("content-length", None)
                
                # Make request to target service
                import json as json_lib
                request_kwargs = {
                    "params": params,
                    "headers": headers
                }
                if body:
                    try:
                        request_kwargs["json"] = json_lib.loads(body.decode())
                    except:
                        request_kwargs["content"] = body
                
                response = await client._make_request(
                    method,
                    f"/{path}",
                    **request_kwargs
                )
                
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            except Exception as e:
                logger.error(f"Gateway routing failed: {e}")
                raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")
        
        @self.app.get("/gateway/services")
        async def list_services():
            """List all registered services."""
            return {
                "services": self.registry.list_services(),
                "routes": self._routes
            }
    
    def _find_target_service(self, path: str) -> Optional[str]:
        """Find target service for path."""
        for route_prefix, service_name in self._routes.items():
            if path.startswith(route_prefix.replace("/api/v1/", "")):
                return service_name
        
        # Default routing
        if path.startswith("move"):
            return "movement-service"
        elif path.startswith("chat"):
            return "chat-service"
        elif path.startswith("trajectory"):
            return "trajectory-service"
        
        return None

