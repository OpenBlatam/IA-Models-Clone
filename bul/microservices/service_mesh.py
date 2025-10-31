"""
BUL Microservices Architecture
=============================

Service mesh implementation for microservices communication and management.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
import httpx
import aiohttp
import consul
import consul.aio

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..security import get_encryption

logger = get_logger(__name__)

class ServiceStatus(str, Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

class ServiceType(str, Enum):
    """Service type enumeration"""
    API_GATEWAY = "api_gateway"
    DOCUMENT_SERVICE = "document_service"
    AGENT_SERVICE = "agent_service"
    ANALYTICS_SERVICE = "analytics_service"
    NOTIFICATION_SERVICE = "notification_service"
    AUTH_SERVICE = "auth_service"
    CACHE_SERVICE = "cache_service"
    DATABASE_SERVICE = "database_service"

@dataclass
class ServiceInstance:
    """Service instance data structure"""
    id: str
    name: str
    service_type: ServiceType
    host: str
    port: int
    status: ServiceStatus
    version: str
    health_check_url: str
    metadata: Dict[str, Any]
    registered_at: datetime
    last_health_check: Optional[datetime] = None
    load_balancer_weight: int = 1
    tags: List[str] = None

@dataclass
class ServiceRoute:
    """Service route configuration"""
    path: str
    service_name: str
    methods: List[str]
    middleware: List[str]
    rate_limit: Optional[Dict[str, Any]] = None
    authentication_required: bool = True
    timeout: int = 30

class ServiceMesh:
    """Service mesh for microservices communication"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.cache_manager = get_cache_manager()
        self.encryption = get_encryption()
        
        # Service registry
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.routes: List[ServiceRoute] = []
        
        # Load balancer
        self.load_balancer = RoundRobinLoadBalancer()
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Service discovery
        self.service_discovery = ServiceDiscovery()
        
        # HTTP clients
        self.http_client: Optional[httpx.AsyncClient] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Health checker
        self.health_checker = ServiceHealthChecker()
        
        # Metrics collector
        self.metrics_collector = ServiceMetricsCollector()
        
        # Initialize services
        self._initialize_services()
    
    async def initialize(self):
        """Initialize service mesh"""
        try:
            # Initialize HTTP clients
            timeout = httpx.Timeout(30.0, connect=10.0)
            self.http_client = httpx.AsyncClient(timeout=timeout)
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            
            # Initialize service discovery
            await self.service_discovery.initialize()
            
            # Initialize health checker
            await self.health_checker.initialize()
            
            # Start background tasks
            asyncio.create_task(self._service_registry_worker())
            asyncio.create_task(self._health_check_worker())
            asyncio.create_task(self._metrics_collection_worker())
            
            self.logger.info("Service mesh initialized successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize service mesh: {e}")
            return False
    
    async def close(self):
        """Close service mesh"""
        try:
            # Close HTTP clients
            if self.http_client:
                await self.http_client.aclose()
            if self.session:
                await self.session.close()
            
            # Close service discovery
            await self.service_discovery.close()
            
            self.logger.info("Service mesh closed")
        
        except Exception as e:
            self.logger.error(f"Error closing service mesh: {e}")
    
    def _initialize_services(self):
        """Initialize default services"""
        # API Gateway
        self._register_service(ServiceInstance(
            id="api-gateway-1",
            name="api-gateway",
            service_type=ServiceType.API_GATEWAY,
            host="localhost",
            port=8000,
            status=ServiceStatus.HEALTHY,
            version="1.0.0",
            health_check_url="/health",
            metadata={"role": "gateway", "priority": "high"},
            registered_at=datetime.now(),
            tags=["gateway", "api"]
        ))
        
        # Document Service
        self._register_service(ServiceInstance(
            id="document-service-1",
            name="document-service",
            service_type=ServiceType.DOCUMENT_SERVICE,
            host="localhost",
            port=8001,
            status=ServiceStatus.HEALTHY,
            version="1.0.0",
            health_check_url="/health",
            metadata={"role": "document_processing", "priority": "high"},
            registered_at=datetime.now(),
            tags=["document", "processing"]
        ))
        
        # Agent Service
        self._register_service(ServiceInstance(
            id="agent-service-1",
            name="agent-service",
            service_type=ServiceType.AGENT_SERVICE,
            host="localhost",
            port=8002,
            status=ServiceStatus.HEALTHY,
            version="1.0.0",
            health_check_url="/health",
            metadata={"role": "agent_management", "priority": "medium"},
            registered_at=datetime.now(),
            tags=["agent", "management"]
        ))
        
        # Analytics Service
        self._register_service(ServiceInstance(
            id="analytics-service-1",
            name="analytics-service",
            service_type=ServiceType.ANALYTICS_SERVICE,
            host="localhost",
            port=8003,
            status=ServiceStatus.HEALTHY,
            version="1.0.0",
            health_check_url="/health",
            metadata={"role": "analytics", "priority": "low"},
            registered_at=datetime.now(),
            tags=["analytics", "metrics"]
        ))
        
        # Initialize routes
        self._initialize_routes()
    
    def _initialize_routes(self):
        """Initialize service routes"""
        self.routes = [
            ServiceRoute(
                path="/api/v1/documents",
                service_name="document-service",
                methods=["GET", "POST", "PUT", "DELETE"],
                middleware=["auth", "rate_limit", "logging"],
                rate_limit={"requests": 100, "window": 60},
                authentication_required=True,
                timeout=30
            ),
            ServiceRoute(
                path="/api/v1/agents",
                service_name="agent-service",
                methods=["GET", "POST", "PUT", "DELETE"],
                middleware=["auth", "rate_limit", "logging"],
                rate_limit={"requests": 200, "window": 60},
                authentication_required=True,
                timeout=15
            ),
            ServiceRoute(
                path="/api/v1/analytics",
                service_name="analytics-service",
                methods=["GET", "POST"],
                middleware=["auth", "rate_limit", "logging"],
                rate_limit={"requests": 50, "window": 60},
                authentication_required=True,
                timeout=20
            ),
            ServiceRoute(
                path="/health",
                service_name="api-gateway",
                methods=["GET"],
                middleware=["logging"],
                authentication_required=False,
                timeout=5
            )
        ]
    
    def _register_service(self, service: ServiceInstance):
        """Register a service instance"""
        if service.name not in self.services:
            self.services[service.name] = []
        
        # Check if service already exists
        existing_services = [s for s in self.services[service.name] if s.id == service.id]
        if existing_services:
            # Update existing service
            existing_services[0] = service
        else:
            # Add new service
            self.services[service.name].append(service)
        
        self.logger.info(f"Registered service: {service.name} ({service.id})")
    
    async def discover_service(self, service_name: str) -> Optional[ServiceInstance]:
        """Discover a healthy service instance"""
        try:
            if service_name not in self.services:
                return None
            
            # Filter healthy services
            healthy_services = [
                service for service in self.services[service_name]
                if service.status == ServiceStatus.HEALTHY
            ]
            
            if not healthy_services:
                return None
            
            # Use load balancer to select service
            selected_service = self.load_balancer.select_service(healthy_services)
            
            # Check circuit breaker
            if not self.circuit_breaker.is_available(selected_service.id):
                # Try next service
                for service in healthy_services:
                    if self.circuit_breaker.is_available(service.id):
                        selected_service = service
                        break
                else:
                    return None
            
            return selected_service
        
        except Exception as e:
            self.logger.error(f"Error discovering service {service_name}: {e}")
            return None
    
    async def call_service(
        self,
        service_name: str,
        method: str,
        path: str,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Call a service through the service mesh"""
        try:
            # Discover service
            service = await self.discover_service(service_name)
            if not service:
                raise HTTPException(status_code=503, detail=f"Service {service_name} not available")
            
            # Build URL
            url = f"http://{service.host}:{service.port}{path}"
            
            # Prepare headers
            request_headers = {
                "Content-Type": "application/json",
                "User-Agent": "BUL-ServiceMesh/1.0",
                "X-Service-Mesh": "true",
                "X-Source-Service": "api-gateway"
            }
            if headers:
                request_headers.update(headers)
            
            # Make request
            start_time = time.time()
            
            try:
                if method.upper() == "GET":
                    response = await self.http_client.get(url, headers=request_headers, timeout=timeout)
                elif method.upper() == "POST":
                    response = await self.http_client.post(url, json=data, headers=request_headers, timeout=timeout)
                elif method.upper() == "PUT":
                    response = await self.http_client.put(url, json=data, headers=request_headers, timeout=timeout)
                elif method.upper() == "DELETE":
                    response = await self.http_client.delete(url, headers=request_headers, timeout=timeout)
                else:
                    raise HTTPException(status_code=405, detail=f"Method {method} not supported")
                
                # Record metrics
                duration = time.time() - start_time
                await self.metrics_collector.record_request(
                    service_name=service_name,
                    method=method,
                    path=path,
                    status_code=response.status_code,
                    duration=duration
                )
                
                # Update circuit breaker
                if response.status_code >= 500:
                    self.circuit_breaker.record_failure(service.id)
                else:
                    self.circuit_breaker.record_success(service.id)
                
                # Return response
                return {
                    "status_code": response.status_code,
                    "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    "headers": dict(response.headers),
                    "service_id": service.id,
                    "duration": duration
                }
            
            except httpx.TimeoutException:
                self.circuit_breaker.record_failure(service.id)
                raise HTTPException(status_code=504, detail="Service request timeout")
            
            except httpx.RequestError as e:
                self.circuit_breaker.record_failure(service.id)
                raise HTTPException(status_code=502, detail=f"Service request failed: {str(e)}")
        
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error calling service {service_name}: {e}")
            raise HTTPException(status_code=500, detail="Internal service mesh error")
    
    async def _service_registry_worker(self):
        """Background worker for service registry management"""
        while True:
            try:
                # Update service registry from service discovery
                discovered_services = await self.service_discovery.get_all_services()
                
                for service in discovered_services:
                    self._register_service(service)
                
                # Clean up old services
                await self._cleanup_old_services()
                
                await asyncio.sleep(30)  # Update every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in service registry worker: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_worker(self):
        """Background worker for health checking"""
        while True:
            try:
                for service_name, services in self.services.items():
                    for service in services:
                        try:
                            # Perform health check
                            is_healthy = await self.health_checker.check_service_health(service)
                            
                            # Update service status
                            if is_healthy:
                                service.status = ServiceStatus.HEALTHY
                            else:
                                service.status = ServiceStatus.UNHEALTHY
                            
                            service.last_health_check = datetime.now()
                        
                        except Exception as e:
                            self.logger.error(f"Health check failed for {service.id}: {e}")
                            service.status = ServiceStatus.UNHEALTHY
                
                await asyncio.sleep(10)  # Health check every 10 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check worker: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_collection_worker(self):
        """Background worker for metrics collection"""
        while True:
            try:
                # Collect and store metrics
                metrics = await self.metrics_collector.collect_metrics()
                
                # Store metrics in cache
                self.cache_manager.set("service_metrics", metrics, ttl=300)
                
                await asyncio.sleep(60)  # Collect metrics every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection worker: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_services(self):
        """Clean up old or unhealthy services"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=5)
            
            for service_name, services in self.services.items():
                # Remove services that haven't been health checked recently
                self.services[service_name] = [
                    service for service in services
                    if service.last_health_check and service.last_health_check > cutoff_time
                ]
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old services: {e}")
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get service mesh metrics"""
        try:
            metrics = {
                "total_services": sum(len(services) for services in self.services.values()),
                "healthy_services": sum(
                    len([s for s in services if s.status == ServiceStatus.HEALTHY])
                    for services in self.services.values()
                ),
                "unhealthy_services": sum(
                    len([s for s in services if s.status == ServiceStatus.UNHEALTHY])
                    for services in self.services.values()
                ),
                "service_types": {
                    service_type.value: len([s for services in self.services.values() for s in services if s.service_type == service_type])
                    for service_type in ServiceType
                },
                "circuit_breaker_status": self.circuit_breaker.get_status(),
                "load_balancer_stats": self.load_balancer.get_stats(),
                "request_metrics": await self.metrics_collector.get_request_metrics()
            }
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error getting service metrics: {e}")
            return {}

class RoundRobinLoadBalancer:
    """Round-robin load balancer for service instances"""
    
    def __init__(self):
        self.current_index = 0
        self.request_counts = {}
    
    def select_service(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Select a service instance using round-robin"""
        if not services:
            return None
        
        # Use round-robin with weights
        total_weight = sum(service.load_balancer_weight for service in services)
        if total_weight == 0:
            return services[0]
        
        # Select service based on weight
        target_weight = self.current_index % total_weight
        current_weight = 0
        
        for service in services:
            current_weight += service.load_balancer_weight
            if current_weight > target_weight:
                self.current_index += 1
                return service
        
        return services[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            "current_index": self.current_index,
            "request_counts": self.request_counts
        }

class CircuitBreaker:
    """Circuit breaker for service fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_counts = {}
        self.last_failure_times = {}
        self.circuit_states = {}  # CLOSED, OPEN, HALF_OPEN
    
    def is_available(self, service_id: str) -> bool:
        """Check if service is available through circuit breaker"""
        if service_id not in self.circuit_states:
            self.circuit_states[service_id] = "CLOSED"
            return True
        
        state = self.circuit_states[service_id]
        
        if state == "CLOSED":
            return True
        elif state == "OPEN":
            # Check if recovery timeout has passed
            if service_id in self.last_failure_times:
                time_since_failure = time.time() - self.last_failure_times[service_id]
                if time_since_failure >= self.recovery_timeout:
                    self.circuit_states[service_id] = "HALF_OPEN"
                    return True
            return False
        elif state == "HALF_OPEN":
            return True
        
        return True
    
    def record_success(self, service_id: str):
        """Record successful request"""
        if service_id in self.failure_counts:
            del self.failure_counts[service_id]
        if service_id in self.last_failure_times:
            del self.last_failure_times[service_id]
        
        self.circuit_states[service_id] = "CLOSED"
    
    def record_failure(self, service_id: str):
        """Record failed request"""
        self.failure_counts[service_id] = self.failure_counts.get(service_id, 0) + 1
        self.last_failure_times[service_id] = time.time()
        
        if self.failure_counts[service_id] >= self.failure_threshold:
            self.circuit_states[service_id] = "OPEN"
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "circuit_states": self.circuit_states,
            "failure_counts": self.failure_counts,
            "last_failure_times": self.last_failure_times
        }

class ServiceDiscovery:
    """Service discovery for finding and registering services"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.consul_client = None
        self.services = {}
    
    async def initialize(self):
        """Initialize service discovery"""
        try:
            # Initialize Consul client (if available)
            # In a real implementation, this would connect to Consul
            self.logger.info("Service discovery initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize service discovery: {e}")
            return False
    
    async def close(self):
        """Close service discovery"""
        try:
            if self.consul_client:
                await self.consul_client.close()
        except Exception as e:
            self.logger.error(f"Error closing service discovery: {e}")
    
    async def get_all_services(self) -> List[ServiceInstance]:
        """Get all discovered services"""
        # In a real implementation, this would query Consul
        return []

class ServiceHealthChecker:
    """Health checker for service instances"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.http_client = None
    
    async def initialize(self):
        """Initialize health checker"""
        try:
            self.http_client = httpx.AsyncClient(timeout=5.0)
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize health checker: {e}")
            return False
    
    async def check_service_health(self, service: ServiceInstance) -> bool:
        """Check if a service is healthy"""
        try:
            if not self.http_client:
                return False
            
            url = f"http://{service.host}:{service.port}{service.health_check_url}"
            response = await self.http_client.get(url, timeout=5.0)
            
            return response.status_code == 200
        
        except Exception as e:
            self.logger.debug(f"Health check failed for {service.id}: {e}")
            return False

class ServiceMetricsCollector:
    """Metrics collector for service mesh"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.request_metrics = {}
        self.service_metrics = {}
    
    async def record_request(
        self,
        service_name: str,
        method: str,
        path: str,
        status_code: int,
        duration: float
    ):
        """Record request metrics"""
        try:
            key = f"{service_name}:{method}:{path}"
            
            if key not in self.request_metrics:
                self.request_metrics[key] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "total_duration": 0.0,
                    "min_duration": float('inf'),
                    "max_duration": 0.0,
                    "avg_duration": 0.0
                }
            
            metrics = self.request_metrics[key]
            metrics["total_requests"] += 1
            metrics["total_duration"] += duration
            metrics["min_duration"] = min(metrics["min_duration"], duration)
            metrics["max_duration"] = max(metrics["max_duration"], duration)
            metrics["avg_duration"] = metrics["total_duration"] / metrics["total_requests"]
            
            if 200 <= status_code < 400:
                metrics["successful_requests"] += 1
            else:
                metrics["failed_requests"] += 1
        
        except Exception as e:
            self.logger.error(f"Error recording request metrics: {e}")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect all metrics"""
        try:
            return {
                "request_metrics": self.request_metrics,
                "service_metrics": self.service_metrics,
                "collected_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return {}
    
    async def get_request_metrics(self) -> Dict[str, Any]:
        """Get request metrics"""
        return self.request_metrics

# Global service mesh
_service_mesh: Optional[ServiceMesh] = None

def get_service_mesh() -> ServiceMesh:
    """Get the global service mesh"""
    global _service_mesh
    if _service_mesh is None:
        _service_mesh = ServiceMesh()
    return _service_mesh

# Service mesh router
service_mesh_router = APIRouter(prefix="/mesh", tags=["Service Mesh"])

@service_mesh_router.get("/services")
async def get_services():
    """Get all registered services"""
    try:
        service_mesh = get_service_mesh()
        return {
            "services": {
                name: [asdict(service) for service in services]
                for name, services in service_mesh.services.items()
            }
        }
    except Exception as e:
        logger.error(f"Error getting services: {e}")
        raise HTTPException(status_code=500, detail="Failed to get services")

@service_mesh_router.get("/services/{service_name}")
async def get_service(service_name: str):
    """Get specific service instances"""
    try:
        service_mesh = get_service_mesh()
        if service_name not in service_mesh.services:
            raise HTTPException(status_code=404, detail="Service not found")
        
        return {
            "service_name": service_name,
            "instances": [asdict(service) for service in service_mesh.services[service_name]]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting service {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service")

@service_mesh_router.get("/metrics")
async def get_service_mesh_metrics():
    """Get service mesh metrics"""
    try:
        service_mesh = get_service_mesh()
        metrics = await service_mesh.get_service_metrics()
        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting service mesh metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@service_mesh_router.post("/services/{service_name}/call")
async def call_service_endpoint(
    service_name: str,
    method: str,
    path: str,
    data: Dict[str, Any] = None
):
    """Call a service through the service mesh"""
    try:
        service_mesh = get_service_mesh()
        result = await service_mesh.call_service(
            service_name=service_name,
            method=method,
            path=path,
            data=data
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calling service {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to call service")


