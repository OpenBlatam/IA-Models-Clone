"""
Integration Manager
===================

Core manager that coordinates all integrated systems and provides unified access.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """System status."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class SystemType(Enum):
    """System types."""
    CONTENT_REDUNDANCY = "content_redundancy"
    BUL = "bul"
    GAMMA_APP = "gamma_app"
    BUSINESS_AGENTS = "business_agents"
    EXPORT_IA = "export_ia"

@dataclass
class SystemInfo:
    """System information."""
    name: str
    type: SystemType
    status: SystemStatus
    endpoint: str
    description: str
    version: str
    last_check: datetime
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntegrationRequest:
    """Integration request."""
    id: str
    source_system: SystemType
    target_system: SystemType
    operation: str
    data: Dict[str, Any]
    created_at: datetime
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class IntegrationManager:
    """Manages integration between all systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.systems: Dict[SystemType, SystemInfo] = {}
        self.integration_requests: Dict[str, IntegrationRequest] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # System configurations
        self.system_configs = {
            SystemType.CONTENT_REDUNDANCY: {
                "name": "Content Redundancy Detector",
                "endpoint": "http://localhost:8001",
                "description": "Detects redundancy in content and analyzes similarity",
                "version": "1.0.0",
                "capabilities": ["content_analysis", "similarity_detection", "quality_assessment"]
            },
            SystemType.BUL: {
                "name": "BUL (Business Unlimited)",
                "endpoint": "http://localhost:8002", 
                "description": "AI-powered document generation for SMEs",
                "version": "1.0.0",
                "capabilities": ["document_generation", "business_analysis", "template_processing"]
            },
            SystemType.GAMMA_APP: {
                "name": "Gamma App",
                "endpoint": "http://localhost:8003",
                "description": "AI-powered content generation system",
                "version": "1.0.0",
                "capabilities": ["content_generation", "presentation_creation", "document_export"]
            },
            SystemType.BUSINESS_AGENTS: {
                "name": "Business Agents",
                "endpoint": "http://localhost:8004",
                "description": "Comprehensive agent system for business areas",
                "version": "1.0.0",
                "capabilities": ["workflow_management", "agent_coordination", "business_automation"]
            },
            SystemType.EXPORT_IA: {
                "name": "Export IA",
                "endpoint": "http://localhost:8005",
                "description": "Advanced document export and analytics",
                "version": "1.0.0",
                "capabilities": ["document_export", "content_analytics", "quality_validation"]
            }
        }
        
        logger.info("Integration Manager initialized")
    
    async def initialize(self):
        """Initialize the integration manager."""
        
        try:
            # Create HTTP client
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Initialize system info
            for system_type, config in self.system_configs.items():
                system_info = SystemInfo(
                    name=config["name"],
                    type=system_type,
                    status=SystemStatus.UNKNOWN,
                    endpoint=config["endpoint"],
                    description=config["description"],
                    version=config["version"],
                    last_check=datetime.now(),
                    capabilities=config["capabilities"]
                )
                self.systems[system_type] = system_info
            
            # Perform initial health checks
            await self._perform_health_checks()
            
            logger.info("Integration Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Integration Manager: {str(e)}")
            raise
    
    async def shutdown(self):
        """Shutdown the integration manager."""
        
        try:
            if self.http_client:
                await self.http_client.aclose()
            
            logger.info("Integration Manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during Integration Manager shutdown: {str(e)}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all systems."""
        
        if not self.http_client:
            return
        
        tasks = []
        for system_type in self.systems:
            task = asyncio.create_task(self._check_system_health(system_type))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_system_health(self, system_type: SystemType):
        """Check health of a specific system."""
        
        system_info = self.systems[system_type]
        
        try:
            start_time = datetime.now()
            
            # Make health check request
            response = await self.http_client.get(f"{system_info.endpoint}/health")
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            if response.status_code == 200:
                system_info.status = SystemStatus.HEALTHY
                system_info.response_time = response_time
                system_info.error_message = None
                
                # Update metadata with response data
                try:
                    health_data = response.json()
                    system_info.metadata.update(health_data)
                except:
                    pass
            else:
                system_info.status = SystemStatus.UNHEALTHY
                system_info.error_message = f"HTTP {response.status_code}"
            
        except Exception as e:
            system_info.status = SystemStatus.OFFLINE
            system_info.error_message = str(e)
            system_info.response_time = None
        
        system_info.last_check = datetime.now()
        
        logger.debug(f"Health check for {system_info.name}: {system_info.status.value}")
    
    async def get_system_status(self, system_type: SystemType) -> SystemInfo:
        """Get status of a specific system."""
        
        if system_type not in self.systems:
            raise ValueError(f"System {system_type} not found")
        
        # Perform fresh health check
        await self._check_system_health(system_type)
        
        return self.systems[system_type]
    
    async def get_all_systems_status(self) -> Dict[SystemType, SystemInfo]:
        """Get status of all systems."""
        
        # Perform fresh health checks
        await self._perform_health_checks()
        
        return self.systems.copy()
    
    async def route_request(
        self,
        system_type: SystemType,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Route request to a specific system."""
        
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")
        
        system_info = self.systems.get(system_type)
        if not system_info:
            raise ValueError(f"System {system_type} not found")
        
        if system_info.status == SystemStatus.OFFLINE:
            raise RuntimeError(f"System {system_info.name} is offline")
        
        try:
            # Prepare request
            url = f"{system_info.endpoint}{endpoint}"
            request_headers = headers or {}
            
            # Make request
            if method.upper() == "GET":
                response = await self.http_client.get(url, headers=request_headers)
            elif method.upper() == "POST":
                response = await self.http_client.post(url, json=data, headers=request_headers)
            elif method.upper() == "PUT":
                response = await self.http_client.put(url, json=data, headers=request_headers)
            elif method.upper() == "DELETE":
                response = await self.http_client.delete(url, headers=request_headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Process response
            if response.status_code < 400:
                try:
                    return response.json()
                except:
                    return {"content": response.text, "status_code": response.status_code}
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Request to {system_info.name} failed: {response.text}"
                )
        
        except httpx.RequestError as e:
            logger.error(f"Request error to {system_info.name}: {str(e)}")
            raise RuntimeError(f"Failed to communicate with {system_info.name}: {str(e)}")
    
    async def create_integration_request(
        self,
        source_system: SystemType,
        target_system: SystemType,
        operation: str,
        data: Dict[str, Any]
    ) -> IntegrationRequest:
        """Create an integration request between systems."""
        
        request_id = str(uuid.uuid4())
        
        request = IntegrationRequest(
            id=request_id,
            source_system=source_system,
            target_system=target_system,
            operation=operation,
            data=data,
            created_at=datetime.now()
        )
        
        self.integration_requests[request_id] = request
        
        # Process the integration request
        await self._process_integration_request(request)
        
        return request
    
    async def _process_integration_request(self, request: IntegrationRequest):
        """Process an integration request."""
        
        try:
            request.status = "processing"
            
            # Route the request to the target system
            result = await self.route_request(
                system_type=request.target_system,
                endpoint=f"/api/v1/{request.operation}",
                method="POST",
                data=request.data
            )
            
            request.result = result
            request.status = "completed"
            
        except Exception as e:
            request.error = str(e)
            request.status = "failed"
            logger.error(f"Integration request {request.id} failed: {str(e)}")
    
    def get_integration_request(self, request_id: str) -> Optional[IntegrationRequest]:
        """Get integration request by ID."""
        return self.integration_requests.get(request_id)
    
    def list_integration_requests(
        self,
        source_system: Optional[SystemType] = None,
        target_system: Optional[SystemType] = None,
        status: Optional[str] = None
    ) -> List[IntegrationRequest]:
        """List integration requests with optional filtering."""
        
        requests = list(self.integration_requests.values())
        
        if source_system:
            requests = [r for r in requests if r.source_system == source_system]
        
        if target_system:
            requests = [r for r in requests if r.target_system == target_system]
        
        if status:
            requests = [r for r in requests if r.status == status]
        
        return sorted(requests, key=lambda x: x.created_at, reverse=True)
    
    async def get_system_capabilities(self, system_type: SystemType) -> List[str]:
        """Get capabilities of a specific system."""
        
        system_info = self.systems.get(system_type)
        if not system_info:
            return []
        
        return system_info.capabilities
    
    async def get_available_operations(self) -> Dict[SystemType, List[str]]:
        """Get available operations for all systems."""
        
        operations = {}
        
        for system_type, system_info in self.systems.items():
            if system_info.status == SystemStatus.HEALTHY:
                try:
                    # Try to get available endpoints from the system
                    response = await self.route_request(
                        system_type=system_type,
                        endpoint="/docs/openapi.json",
                        method="GET"
                    )
                    
                    # Parse OpenAPI spec to extract operations
                    paths = response.get("paths", {})
                    system_operations = []
                    
                    for path, methods in paths.items():
                        for method, details in methods.items():
                            if method.upper() in ["GET", "POST", "PUT", "DELETE"]:
                                operation_id = details.get("operationId", f"{method.upper()} {path}")
                                system_operations.append(operation_id)
                    
                    operations[system_type] = system_operations
                    
                except Exception as e:
                    logger.warning(f"Could not get operations for {system_info.name}: {str(e)}")
                    operations[system_type] = system_info.capabilities
        
        return operations
    
    async def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        
        total_requests = len(self.integration_requests)
        completed_requests = len([r for r in self.integration_requests.values() if r.status == "completed"])
        failed_requests = len([r for r in self.integration_requests.values() if r.status == "failed"])
        processing_requests = len([r for r in self.integration_requests.values() if r.status == "processing"])
        
        # System status distribution
        system_status_distribution = {}
        for system_info in self.systems.values():
            status = system_info.status.value
            system_status_distribution[status] = system_status_distribution.get(status, 0) + 1
        
        # Request distribution by system
        request_distribution = {}
        for request in self.integration_requests.values():
            target = request.target_system.value
            request_distribution[target] = request_distribution.get(target, 0) + 1
        
        return {
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "processing_requests": processing_requests,
            "success_rate": completed_requests / total_requests if total_requests > 0 else 0,
            "system_status_distribution": system_status_distribution,
            "request_distribution": request_distribution,
            "total_systems": len(self.systems),
            "healthy_systems": len([s for s in self.systems.values() if s.status == SystemStatus.HEALTHY])
        }

