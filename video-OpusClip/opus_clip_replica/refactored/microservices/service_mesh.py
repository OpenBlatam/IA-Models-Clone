"""
Microservices Architecture for Opus Clip

Advanced microservices implementation with:
- Service mesh architecture
- API Gateway
- Service discovery
- Load balancing
- Circuit breakers
- Distributed tracing
- Event-driven communication
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Callable
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import httpx
import redis
import consul
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import opentelemetry
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
import uvicorn

logger = structlog.get_logger("service_mesh")

class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class CircuitState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class ServiceConfig:
    """Service configuration."""
    name: str
    host: str
    port: int
    version: str = "1.0.0"
    health_check_path: str = "/health"
    timeout: float = 30.0
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

@dataclass
class ServiceInstance:
    """Service instance information."""
    id: str
    name: str
    host: str
    port: int
    version: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CircuitBreaker:
    """Circuit breaker implementation."""
    service_name: str
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: CircuitState = CircuitState.CLOSED
    threshold: int = 5
    timeout: float = 60.0

class ServiceRegistry:
    """Service registry for service discovery."""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul_client = consul.Consul(host=consul_host, port=consul_port)
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.logger = structlog.get_logger("service_registry")
    
    async def register_service(self, service: ServiceInstance):
        """Register a service instance."""
        try:
            service_id = f"{service.name}-{service.id}"
            
            # Register with Consul
            self.consul_client.agent.service.register(
                name=service.name,
                service_id=service_id,
                address=service.host,
                port=service.port,
                check=consul.Check.http(
                    f"http://{service.host}:{service.port}{service.metadata.get('health_check_path', '/health')}",
                    interval="10s"
                ),
                meta=service.metadata
            )
            
            # Add to local registry
            if service.name not in self.services:
                self.services[service.name] = []
            self.services[service.name].append(service)
            
            self.logger.info(f"Registered service: {service.name} at {service.host}:{service.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to register service {service.name}: {e}")
            raise
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover service instances."""
        try:
            # Get from Consul
            _, services = self.consul_client.health.service(service_name, passing=True)
            
            instances = []
            for service in services:
                instance = ServiceInstance(
                    id=service['Service']['ID'],
                    name=service['Service']['Service'],
                    host=service['Service']['Address'],
                    port=service['Service']['Port'],
                    version=service['Service']['Meta'].get('version', '1.0.0'),
                    metadata=service['Service']['Meta']
                )
                instances.append(instance)
            
            # Update local registry
            self.services[service_name] = instances
            
            return instances
            
        except Exception as e:
            self.logger.error(f"Failed to discover services for {service_name}: {e}")
            return self.services.get(service_name, [])
    
    async def deregister_service(self, service_name: str, service_id: str):
        """Deregister a service instance."""
        try:
            self.consul_client.agent.service.deregister(service_id)
            
            # Remove from local registry
            if service_name in self.services:
                self.services[service_name] = [
                    s for s in self.services[service_name] if s.id != service_id
                ]
            
            self.logger.info(f"Deregistered service: {service_name} - {service_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to deregister service {service_name}: {e}")
            raise

class LoadBalancer:
    """Load balancer for service instances."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.current_index = 0
        self.logger = structlog.get_logger("load_balancer")
    
    def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select a service instance using the configured strategy."""
        if not instances:
            return None
        
        # Filter healthy instances
        healthy_instances = [i for i in instances if i.status == ServiceStatus.HEALTHY]
        
        if not healthy_instances:
            # Fallback to any instance if no healthy ones
            healthy_instances = instances
        
        if self.strategy == "round_robin":
            instance = healthy_instances[self.current_index % len(healthy_instances)]
            self.current_index += 1
            return instance
        
        elif self.strategy == "random":
            import random
            return random.choice(healthy_instances)
        
        elif self.strategy == "least_connections":
            # Simple implementation - in practice, track connection counts
            return healthy_instances[0]
        
        else:
            return healthy_instances[0]

class CircuitBreakerManager:
    """Circuit breaker manager for fault tolerance."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = structlog.get_logger("circuit_breaker")
    
    def get_circuit_breaker(self, service_name: str, threshold: int = 5, timeout: float = 60.0) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                service_name=service_name,
                threshold=threshold,
                timeout=timeout
            )
        return self.circuit_breakers[service_name]
    
    def record_success(self, service_name: str):
        """Record a successful call."""
        if service_name in self.circuit_breakers:
            circuit = self.circuit_breakers[service_name]
            circuit.failure_count = 0
            circuit.state = CircuitState.CLOSED
    
    def record_failure(self, service_name: str):
        """Record a failed call."""
        if service_name in self.circuit_breakers:
            circuit = self.circuit_breakers[service_name]
            circuit.failure_count += 1
            circuit.last_failure_time = datetime.now()
            
            if circuit.failure_count >= circuit.threshold:
                circuit.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker opened for {service_name}")
    
    def is_call_allowed(self, service_name: str) -> bool:
        """Check if a call is allowed through the circuit breaker."""
        if service_name not in self.circuit_breakers:
            return True
        
        circuit = self.circuit_breakers[service_name]
        
        if circuit.state == CircuitState.CLOSED:
            return True
        
        elif circuit.state == CircuitState.OPEN:
            if circuit.last_failure_time and \
               datetime.now() - circuit.last_failure_time > timedelta(seconds=circuit.timeout):
                circuit.state = CircuitState.HALF_OPEN
                return True
            return False
        
        elif circuit.state == CircuitState.HALF_OPEN:
            return True
        
        return False

class ServiceMesh:
    """Main service mesh implementation."""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.registry = ServiceRegistry(consul_host, consul_port)
        self.load_balancer = LoadBalancer()
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.logger = structlog.get_logger("service_mesh")
        
        # Metrics
        self.request_counter = Counter('service_requests_total', 'Total service requests', ['service', 'method', 'status'])
        self.request_duration = Histogram('service_request_duration_seconds', 'Service request duration', ['service', 'method'])
        self.active_connections = Gauge('service_active_connections', 'Active service connections', ['service'])
        
        # Tracing
        self._setup_tracing()
    
    def _setup_tracing(self):
        """Setup distributed tracing."""
        try:
            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
            )
            
            # Set up tracer provider
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            # Add span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Instrument FastAPI and HTTP client
            FastAPIInstrumentor.instrument_app
            HTTPXClientInstrumentor.instrument()
            
            self.logger.info("Distributed tracing configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup tracing: {e}")
    
    async def register_service(self, service_config: ServiceConfig):
        """Register a service in the mesh."""
        service_instance = ServiceInstance(
            id=str(uuid.uuid4()),
            name=service_config.name,
            host=service_config.host,
            port=service_config.port,
            version=service_config.version,
            metadata={
                "health_check_path": service_config.health_check_path,
                "timeout": service_config.timeout,
                "retry_attempts": service_config.retry_attempts
            }
        )
        
        await self.registry.register_service(service_instance)
    
    async def discover_service(self, service_name: str) -> List[ServiceInstance]:
        """Discover service instances."""
        return await self.registry.discover_services(service_name)
    
    async def call_service(self, service_name: str, method: str, path: str, 
                          data: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Call a service through the mesh."""
        start_time = time.time()
        
        try:
            # Check circuit breaker
            if not self.circuit_breaker_manager.is_call_allowed(service_name):
                raise HTTPException(status_code=503, detail="Service circuit breaker is open")
            
            # Discover service instances
            instances = await self.discover_service(service_name)
            if not instances:
                raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
            
            # Select instance using load balancer
            instance = self.load_balancer.select_instance(instances)
            if not instance:
                raise HTTPException(status_code=503, detail="No healthy service instances available")
            
            # Make HTTP request
            url = f"http://{instance.host}:{instance.port}{path}"
            
            with self.request_duration.labels(service=service_name, method=method).time():
                if method.upper() == "GET":
                    response = await self.http_client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await self.http_client.post(url, json=data, headers=headers)
                elif method.upper() == "PUT":
                    response = await self.http_client.put(url, json=data, headers=headers)
                elif method.upper() == "DELETE":
                    response = await self.http_client.delete(url, headers=headers)
                else:
                    raise HTTPException(status_code=405, detail=f"Method {method} not supported")
            
            # Record success
            self.circuit_breaker_manager.record_success(service_name)
            
            # Update metrics
            self.request_counter.labels(
                service=service_name,
                method=method,
                status=response.status_code
            ).inc()
            
            return {
                "status_code": response.status_code,
                "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                "headers": dict(response.headers),
                "service_instance": {
                    "id": instance.id,
                    "host": instance.host,
                    "port": instance.port
                }
            }
            
        except httpx.HTTPStatusError as e:
            # Record failure
            self.circuit_breaker_manager.record_failure(service_name)
            
            # Update metrics
            self.request_counter.labels(
                service=service_name,
                method=method,
                status=e.response.status_code
            ).inc()
            
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
            
        except Exception as e:
            # Record failure
            self.circuit_breaker_manager.record_failure(service_name)
            
            # Update metrics
            self.request_counter.labels(
                service=service_name,
                method=method,
                status=500
            ).inc()
            
            self.logger.error(f"Service call failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            duration = time.time() - start_time
            self.request_duration.labels(service=service_name, method=method).observe(duration)
    
    async def health_check(self, service_name: str) -> Dict[str, Any]:
        """Perform health check on a service."""
        try:
            instances = await self.discover_service(service_name)
            healthy_instances = [i for i in instances if i.status == ServiceStatus.HEALTHY]
            
            return {
                "service": service_name,
                "total_instances": len(instances),
                "healthy_instances": len(healthy_instances),
                "circuit_breaker_state": self.circuit_breaker_manager.circuit_breakers.get(service_name, {}).get("state", "unknown"),
                "instances": [
                    {
                        "id": i.id,
                        "host": i.host,
                        "port": i.port,
                        "status": i.status.value
                    }
                    for i in instances
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed for {service_name}: {e}")
            return {
                "service": service_name,
                "error": str(e),
                "status": "unhealthy"
            }
    
    async def get_mesh_status(self) -> Dict[str, Any]:
        """Get overall mesh status."""
        try:
            services = list(self.registry.services.keys())
            circuit_breakers = {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breaker_manager.circuit_breakers.items()
            }
            
            return {
                "mesh_status": "healthy",
                "registered_services": services,
                "circuit_breakers": circuit_breakers,
                "load_balancer_strategy": self.load_balancer.strategy,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get mesh status: {e}")
            return {
                "mesh_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.http_client.aclose()
        self.logger.info("Service mesh cleaned up")

# API Gateway
class APIGateway:
    """API Gateway for the service mesh."""
    
    def __init__(self, service_mesh: ServiceMesh):
        self.service_mesh = service_mesh
        self.app = FastAPI(title="Opus Clip API Gateway", version="1.0.0")
        self.logger = structlog.get_logger("api_gateway")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API Gateway routes."""
        
        @self.app.get("/health")
        async def gateway_health():
            """Gateway health check."""
            return {"status": "healthy", "gateway": "api-gateway"}
        
        @self.app.get("/mesh/status")
        async def mesh_status():
            """Get service mesh status."""
            return await self.service_mesh.get_mesh_status()
        
        @self.app.get("/mesh/services/{service_name}/health")
        async def service_health(service_name: str):
            """Get service health."""
            return await self.service_mesh.health_check(service_name)
        
        @self.app.post("/mesh/services/{service_name}/call")
        async def call_service(
            service_name: str,
            method: str,
            path: str,
            data: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None
        ):
            """Call a service through the mesh."""
            return await self.service_mesh.call_service(service_name, method, path, data, headers)
        
        # Video processing routes
        @self.app.post("/api/video/analyze")
        async def analyze_video(request: Dict[str, Any]):
            """Analyze video through video service."""
            return await self.service_mesh.call_service(
                "video-service",
                "POST",
                "/analyze",
                data=request
            )
        
        @self.app.post("/api/video/extract")
        async def extract_clips(request: Dict[str, Any]):
            """Extract clips through video service."""
            return await self.service_mesh.call_service(
                "video-service",
                "POST",
                "/extract",
                data=request
            )
        
        # Analytics routes
        @self.app.get("/api/analytics/metrics")
        async def get_metrics():
            """Get analytics metrics."""
            return await self.service_mesh.call_service(
                "analytics-service",
                "GET",
                "/metrics"
            )
        
        # User management routes
        @self.app.post("/api/users/login")
        async def user_login(request: Dict[str, Any]):
            """User login through user service."""
            return await self.service_mesh.call_service(
                "user-service",
                "POST",
                "/login",
                data=request
            )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API Gateway."""
        uvicorn.run(self.app, host=host, port=port)

# Event Bus for Event-Driven Communication
class EventBus:
    """Event bus for microservices communication."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.subscribers: Dict[str, List[Callable]] = {}
        self.logger = structlog.get_logger("event_bus")
    
    async def publish(self, topic: str, event: Dict[str, Any]):
        """Publish an event to a topic."""
        try:
            event_data = {
                "id": str(uuid.uuid4()),
                "topic": topic,
                "data": event,
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish to Redis
            self.redis_client.publish(topic, json.dumps(event_data))
            
            self.logger.info(f"Published event to topic {topic}: {event_data['id']}")
            
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
            raise
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        
        self.logger.info(f"Subscribed to topic: {topic}")
    
    async def start_listening(self):
        """Start listening for events."""
        pubsub = self.redis_client.pubsub()
        
        # Subscribe to all topics
        for topic in self.subscribers.keys():
            pubsub.subscribe(topic)
        
        # Listen for messages
        for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    event_data = json.loads(message['data'])
                    topic = event_data['topic']
                    
                    # Call subscribers
                    if topic in self.subscribers:
                        for callback in self.subscribers[topic]:
                            try:
                                await callback(event_data)
                            except Exception as e:
                                self.logger.error(f"Error in event callback: {e}")
                
                except Exception as e:
                    self.logger.error(f"Error processing event: {e}")

# Example usage
async def main():
    """Example usage of the service mesh."""
    # Initialize service mesh
    mesh = ServiceMesh()
    
    # Register services
    video_service_config = ServiceConfig(
        name="video-service",
        host="localhost",
        port=8001,
        version="1.0.0"
    )
    
    analytics_service_config = ServiceConfig(
        name="analytics-service",
        host="localhost",
        port=8002,
        version="1.0.0"
    )
    
    await mesh.register_service(video_service_config)
    await mesh.register_service(analytics_service_config)
    
    # Initialize API Gateway
    gateway = APIGateway(mesh)
    
    # Start event bus
    event_bus = EventBus()
    
    # Subscribe to events
    async def video_processed_callback(event):
        print(f"Video processed: {event['data']}")
    
    await event_bus.subscribe("video.processed", video_processed_callback)
    
    # Start listening for events
    asyncio.create_task(event_bus.start_listening())
    
    # Run API Gateway
    gateway.run()

if __name__ == "__main__":
    asyncio.run(main())


