"""
API Gateway for service orchestration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from .core import BaseService, get_service_manager
from .communication import get_message_bus, ServiceClient
from .discovery import get_service_discovery

logger = logging.getLogger(__name__)


class APIGateway(BaseService):
    """API Gateway for orchestrating microservices."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        super().__init__("api-gateway", "1.0.0", host, port)
        self.app = FastAPI(
            title="Export IA API Gateway",
            description="Microservices API Gateway",
            version="2.0.0"
        )
        self.message_bus = get_message_bus()
        self.service_discovery = get_service_discovery()
        self.service_client = ServiceClient(self.message_bus, "api-gateway")
        
        # Add dependencies
        self.add_dependency("export-service")
        self.add_dependency("quality-service")
        self.add_dependency("task-service")
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self) -> None:
        """Setup middleware for the API gateway."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = asyncio.get_event_loop().time()
            
            response = await call_next(request)
            
            process_time = asyncio.get_event_loop().time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            return response
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "name": "Export IA API Gateway",
                "version": "2.0.0",
                "status": "running",
                "services": await self._get_available_services()
            }
        
        @self.app.get("/health")
        async def health():
            return await self.health_check()
        
        @self.app.get("/services")
        async def list_services():
            """List all available services."""
            return await self.service_discovery.get_healthy_services()
        
        @self.app.get("/services/{service_name}")
        async def get_service_info(service_name: str):
            """Get information about a specific service."""
            services = await self.service_discovery.discover_services(service_name)
            if not services:
                raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
            return services[0]
        
        # Export service routes
        @self.app.post("/export")
        async def export_document(request: Request):
            """Export a document."""
            try:
                data = await request.json()
                content = data.get("content")
                config = data.get("config", {})
                
                if not content:
                    raise HTTPException(status_code=400, detail="Content is required")
                
                # Call export service
                task_id = await self.service_client.call_export_service(content, config)
                
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "message": "Export task created"
                }
                
            except Exception as e:
                logger.error(f"Export request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/export/{task_id}/status")
        async def get_export_status(task_id: str):
            """Get export task status."""
            try:
                # Call task service
                response = await self.service_client.call_service(
                    "task-service",
                    "status",
                    {"task_id": task_id}
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Status request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/export/{task_id}/download")
        async def download_export(task_id: str):
            """Download exported file."""
            try:
                # Call task service
                response = await self.service_client.call_service(
                    "task-service",
                    "download",
                    {"task_id": task_id}
                )
                
                if "file_path" not in response:
                    raise HTTPException(status_code=404, detail="File not found")
                
                # Return file response
                from fastapi.responses import FileResponse
                return FileResponse(
                    path=response["file_path"],
                    filename=response.get("filename", "export.pdf")
                )
                
            except Exception as e:
                logger.error(f"Download request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Quality service routes
        @self.app.post("/validate")
        async def validate_content(request: Request):
            """Validate document content."""
            try:
                data = await request.json()
                content = data.get("content")
                config = data.get("config", {})
                
                if not content:
                    raise HTTPException(status_code=400, detail="Content is required")
                
                # Call quality service
                result = await self.service_client.call_quality_service(content, config)
                
                return result
                
            except Exception as e:
                logger.error(f"Validation request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Statistics routes
        @self.app.get("/statistics")
        async def get_statistics():
            """Get system statistics."""
            try:
                # Call task service for statistics
                response = await self.service_client.call_service(
                    "task-service",
                    "statistics",
                    {}
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Statistics request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Service management routes
        @self.app.post("/services/{service_name}/restart")
        async def restart_service(service_name: str):
            """Restart a service."""
            try:
                service_manager = get_service_manager()
                services = await self.service_discovery.discover_services(service_name)
                
                if not services:
                    raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
                
                # Restart service (implementation depends on service manager)
                return {"message": f"Service {service_name} restart requested"}
                
            except Exception as e:
                logger.error(f"Service restart failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_available_services(self) -> List[str]:
        """Get list of available services."""
        healthy_services = await self.service_discovery.get_healthy_services()
        return list(healthy_services.keys())
    
    async def _start(self) -> None:
        """Start the API gateway."""
        # Start message bus
        await self.message_bus.start()
        
        # Start service discovery
        await self.service_discovery.start()
        
        # Register with service discovery
        await self.service_discovery.register_service(
            name=self.name,
            host=self.host,
            port=self.port,
            health_url="/health",
            api_url="/"
        )
        
        # Start the FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Run server in background
        self._server_task = asyncio.create_task(server.serve())
        
        logger.info(f"API Gateway started on {self.host}:{self.port}")
    
    async def _stop(self) -> None:
        """Stop the API gateway."""
        # Stop server
        if hasattr(self, '_server_task'):
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        # Stop service discovery
        await self.service_discovery.stop()
        
        # Stop message bus
        await self.message_bus.stop()
        
        logger.info("API Gateway stopped")


class LoadBalancer:
    """Load balancer for distributing requests across service instances."""
    
    def __init__(self, service_discovery: Any):
        self.service_discovery = service_discovery
        self.round_robin_counters: Dict[str, int] = {}
    
    async def get_service_endpoint(self, service_name: str, strategy: str = "round_robin") -> Optional[str]:
        """Get a service endpoint using the specified strategy."""
        services = await self.service_discovery.discover_services(service_name)
        
        if not services:
            return None
        
        if strategy == "round_robin":
            return self._round_robin_selection(service_name, services)
        elif strategy == "random":
            import random
            service = random.choice(services)
            return f"http://{service.host}:{service.port}"
        elif strategy == "first":
            service = services[0]
            return f"http://{service.host}:{service.port}"
        else:
            # Default to round robin
            return self._round_robin_selection(service_name, services)
    
    def _round_robin_selection(self, service_name: str, services: List[Any]) -> str:
        """Round robin selection of service."""
        if service_name not in self.round_robin_counters:
            self.round_robin_counters[service_name] = 0
        
        index = self.round_robin_counters[service_name] % len(services)
        self.round_robin_counters[service_name] += 1
        
        service = services[index]
        return f"http://{service.host}:{service.port}"


# Global API gateway instance
_api_gateway: Optional[APIGateway] = None


def get_api_gateway() -> APIGateway:
    """Get the global API gateway instance."""
    global _api_gateway
    if _api_gateway is None:
        _api_gateway = APIGateway()
    return _api_gateway




