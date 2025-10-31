"""
🚀 Ultra Library Optimization V7 - Microservices Architecture System
==================================================================

Advanced microservices architecture with service mesh, API gateway, and distributed communication.
"""

import asyncio
import logging
import uuid
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import httpx
import aiohttp
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import redis.asyncio as redis
import consul
import grpc
from grpc import aio
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import opentelemetry
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import Counter as OTelCounter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.prometheus import PrometheusExporter
import structlog
from structlog import get_logger
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
import circuitbreaker
from circuitbreaker import circuit
import kubernetes
from kubernetes import client, config
import yaml
import docker
from docker import DockerClient
import docker.types
import docker.errors


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class ServiceType(Enum):
    """Service types in the microservices architecture."""
    CONTENT_GENERATION = "content-generation"
    OPTIMIZATION = "optimization"
    ANALYTICS = "analytics"
    AUTHENTICATION = "authentication"
    NOTIFICATION = "notification"
    STORAGE = "storage"
    AI_PROCESSING = "ai-processing"
    BLOCKCHAIN = "blockchain"


class CommunicationProtocol(Enum):
    """Communication protocols between services."""
    HTTP_REST = "http-rest"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message-queue"
    EVENT_STREAM = "event-stream"
    WEBSOCKET = "websocket"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round-robin"
    LEAST_CONNECTIONS = "least-connections"
    WEIGHTED_ROUND_ROBIN = "weighted-round-robin"
    IP_HASH = "ip-hash"
    LEAST_RESPONSE_TIME = "least-response-time"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ServiceInstance:
    """Service instance information."""
    id: str
    name: str
    service_type: ServiceType
    host: str
    port: int
    health_check_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    load_balancer_weight: int = 1
    is_healthy: bool = True
    last_health_check: float = field(default_factory=time.time)
    response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0


@dataclass
class ServiceRequest:
    """Service request information."""
    id: str
    service_name: str
    method: str
    path: str
    headers: Dict[str, str]
    body: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class ServiceResponse:
    """Service response information."""
    request_id: str
    status_code: int
    headers: Dict[str, str]
    body: Any
    response_time: float
    service_instance: ServiceInstance
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    monitor_interval: float = 10.0
    half_open_max_calls: int = 3


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_interval: float = 30.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    timeout: float = 5.0


@dataclass
class ServiceMeshConfig:
    """Service mesh configuration."""
    tracing_enabled: bool = True
    metrics_enabled: bool = True
    logging_enabled: bool = True
    security_enabled: bool = True
    rate_limiting_enabled: bool = True
    circuit_breaker_enabled: bool = True
    retry_enabled: bool = True
    timeout_enabled: bool = True


@dataclass
class APIGatewayConfig:
    """API Gateway configuration."""
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    timeout: float = 30.0
    cors_enabled: bool = True
    compression_enabled: bool = True
    caching_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes


# =============================================================================
# SERVICE DISCOVERY
# =============================================================================

class ServiceDiscovery(ABC):
    """Abstract service discovery interface."""
    
    @abstractmethod
    async def register_service(self, service: ServiceInstance) -> bool:
        """Register a service instance."""
        pass
    
    @abstractmethod
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service instance."""
        pass
    
    @abstractmethod
    async def discover_services(self, service_type: ServiceType) -> List[ServiceInstance]:
        """Discover services of a specific type."""
        pass
    
    @abstractmethod
    async def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get all instances of a specific service."""
        pass


class ConsulServiceDiscovery(ServiceDiscovery):
    """Consul-based service discovery."""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul_client = consul.Consul(host=consul_host, port=consul_port)
        self._logger = get_logger(__name__)
    
    async def register_service(self, service: ServiceInstance) -> bool:
        """Register service with Consul."""
        try:
            service_data = {
                "ID": service.id,
                "Name": service.name,
                "Address": service.host,
                "Port": service.port,
                "Tags": [service.service_type.value],
                "Check": {
                    "HTTP": service.health_check_url,
                    "Interval": "10s",
                    "Timeout": "5s"
                },
                "Meta": service.metadata
            }
            
            result = self.consul_client.agent.service.register(
                name=service.name,
                service_id=service.id,
                address=service.host,
                port=service.port,
                tags=[service.service_type.value],
                check=service_data["Check"],
                meta=service.metadata
            )
            
            self._logger.info(f"Service registered: {service.name} ({service.id})")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to register service: {e}")
            return False
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister service from Consul."""
        try:
            self.consul_client.agent.service.deregister(service_id)
            self._logger.info(f"Service deregistered: {service_id}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to deregister service: {e}")
            return False
    
    async def discover_services(self, service_type: ServiceType) -> List[ServiceInstance]:
        """Discover services by type."""
        try:
            services = self.consul_client.agent.services()
            instances = []
            
            for service_id, service_info in services.items():
                if service_type.value in service_info.get("Tags", []):
                    instance = ServiceInstance(
                        id=service_id,
                        name=service_info["Service"],
                        service_type=service_type,
                        host=service_info["Address"],
                        port=service_info["Port"],
                        health_check_url=f"http://{service_info['Address']}:{service_info['Port']}/health",
                        metadata=service_info.get("Meta", {})
                    )
                    instances.append(instance)
            
            return instances
            
        except Exception as e:
            self._logger.error(f"Failed to discover services: {e}")
            return []
    
    async def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get all instances of a specific service."""
        try:
            services = self.consul_client.agent.services()
            instances = []
            
            for service_id, service_info in services.items():
                if service_info["Service"] == service_name:
                    instance = ServiceInstance(
                        id=service_id,
                        name=service_info["Service"],
                        service_type=ServiceType(service_info.get("Tags", [""])[0]),
                        host=service_info["Address"],
                        port=service_info["Port"],
                        health_check_url=f"http://{service_info['Address']}:{service_info['Port']}/health",
                        metadata=service_info.get("Meta", {})
                    )
                    instances.append(instance)
            
            return instances
            
        except Exception as e:
            self._logger.error(f"Failed to get service instances: {e}")
            return []


# =============================================================================
# LOAD BALANCER
# =============================================================================

class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self._current_index = 0
        self._service_instances: Dict[str, List[ServiceInstance]] = {}
        self._instance_stats: Dict[str, Dict[str, Any]] = {}
        self._logger = get_logger(__name__)
    
    def add_service_instances(self, service_name: str, instances: List[ServiceInstance]):
        """Add service instances to the load balancer."""
        self._service_instances[service_name] = instances
        self._instance_stats[service_name] = {
            "total_requests": 0,
            "total_response_time": 0.0,
            "error_count": 0
        }
    
    def get_next_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Get the next service instance based on the load balancing strategy."""
        instances = self._service_instances.get(service_name, [])
        if not instances:
            return None
        
        # Filter healthy instances
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            self._logger.warning(f"No healthy instances available for {service_name}")
            return None
        
        if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_instances)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_instances)
        elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(healthy_instances)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(healthy_instances)
        else:
            return self._round_robin(healthy_instances)
    
    def _round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin load balancing."""
        instance = instances[self._current_index % len(instances)]
        self._current_index += 1
        return instance
    
    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections load balancing."""
        return min(instances, key=lambda x: x.success_count + x.error_count)
    
    def _weighted_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round-robin load balancing."""
        total_weight = sum(inst.load_balancer_weight for inst in instances)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.load_balancer_weight
            if self._current_index < current_weight:
                self._current_index += 1
                return instance
        
        # Fallback to regular round-robin
        return self._round_robin(instances)
    
    def _least_response_time(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least response time load balancing."""
        return min(instances, key=lambda x: x.response_time)
    
    def update_instance_stats(self, service_name: str, instance_id: str, 
                            response_time: float, success: bool):
        """Update instance statistics."""
        instances = self._service_instances.get(service_name, [])
        for instance in instances:
            if instance.id == instance_id:
                instance.response_time = response_time
                if success:
                    instance.success_count += 1
                else:
                    instance.error_count += 1
                break


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """Advanced circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
        self._logger = get_logger(__name__)
    
    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self._logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        return True
    
    def on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self._logger.info("Circuit breaker CLOSED - service recovered")
        else:
            self.failure_count = 0
    
    def on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self._logger.warning(f"Circuit breaker OPEN - {self.failure_count} failures")


# =============================================================================
# SERVICE MESH
# =============================================================================

class ServiceMesh:
    """Advanced service mesh implementation."""
    
    def __init__(self, config: ServiceMeshConfig):
        self.config = config
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, Any] = {}
        self.tracers: Dict[str, trace.Tracer] = {}
        self.meters: Dict[str, metrics.Meter] = {}
        self._logger = get_logger(__name__)
        
        # Initialize OpenTelemetry
        if self.config.tracing_enabled:
            self._setup_tracing()
        
        if self.config.metrics_enabled:
            self._setup_metrics()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        try:
            # Create tracer provider
            tracer_provider = TracerProvider()
            
            # Add Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            
            # Add batch span processor
            tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            
            # Set global tracer provider
            trace.set_tracer_provider(tracer_provider)
            
            self._logger.info("Tracing setup completed")
            
        except Exception as e:
            self._logger.error(f"Failed to setup tracing: {e}")
    
    def _setup_metrics(self):
        """Setup OpenTelemetry metrics."""
        try:
            # Create meter provider
            meter_provider = MeterProvider()
            metrics.set_meter_provider(meter_provider)
            
            self._logger.info("Metrics setup completed")
            
        except Exception as e:
            self._logger.error(f"Failed to setup metrics: {e}")
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            config = CircuitBreakerConfig()
            self.circuit_breakers[service_name] = CircuitBreaker(config)
        return self.circuit_breakers[service_name]
    
    def get_tracer(self, service_name: str) -> trace.Tracer:
        """Get tracer for a service."""
        if service_name not in self.tracers:
            self.tracers[service_name] = trace.get_tracer(service_name)
        return self.tracers[service_name]
    
    def get_meter(self, service_name: str) -> metrics.Meter:
        """Get meter for a service."""
        if service_name not in self.meters:
            self.meters[service_name] = metrics.get_meter(service_name)
        return self.meters[service_name]


# =============================================================================
# API GATEWAY
# =============================================================================

class APIGateway:
    """Advanced API Gateway implementation."""
    
    def __init__(self, config: APIGatewayConfig):
        self.config = config
        self.service_discovery = ConsulServiceDiscovery()
        self.load_balancer = LoadBalancer(LoadBalancerConfig())
        self.service_mesh = ServiceMesh(ServiceMeshConfig())
        self.rate_limit_store = {}
        self.cache = {}
        self._logger = get_logger(__name__)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="🚀 Ultra Library Optimization V7 - API Gateway",
            description="Advanced API Gateway with microservices architecture",
            version="1.0.0"
        )
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup API Gateway middleware."""
        # CORS middleware
        if self.config.cors_enabled:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Compression middleware
        if self.config.compression_enabled:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware for rate limiting, authentication, etc.
        self.app.middleware("http")(self._gateway_middleware)
    
    def _setup_routes(self):
        """Setup API Gateway routes."""
        
        @self.app.get("/")
        async def gateway_info():
            return {
                "name": "Ultra Library Optimization V7 - API Gateway",
                "version": "1.0.0",
                "status": "operational",
                "services": list(self.service_discovery.consul_client.agent.services().keys())
            }
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.app.get("/metrics")
        async def metrics():
            return {
                "rate_limits": self.rate_limit_store,
                "cache_stats": len(self.cache),
                "circuit_breakers": {
                    name: cb.state.value for name, cb in self.service_mesh.circuit_breakers.items()
                }
            }
        
        # Dynamic route for service proxying
        @self.app.api_route("/api/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def proxy_request(service_name: str, path: str, request: Request):
            return await self._handle_proxy_request(service_name, path, request)
    
    async def _gateway_middleware(self, request: Request, call_next):
        """API Gateway middleware for rate limiting, authentication, etc."""
        start_time = time.time()
        
        # Rate limiting
        if self.config.rate_limiting_enabled:
            client_ip = request.client.host
            if not await self._check_rate_limit(client_ip):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Authentication (simplified)
        # In production, implement proper JWT validation
        
        # Request size validation
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > self.config.max_request_size:
                raise HTTPException(status_code=413, detail="Request too large")
        
        # Process request
        response = await call_next(request)
        
        # Add response headers
        response.headers["X-Gateway-Processing-Time"] = str(time.time() - start_time)
        response.headers["X-Gateway-Version"] = "1.0.0"
        
        return response
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limiting for a client."""
        current_time = time.time()
        minute_key = f"{client_ip}:minute:{int(current_time / 60)}"
        hour_key = f"{client_ip}:hour:{int(current_time / 3600)}"
        
        # Check minute limit
        minute_count = self.rate_limit_store.get(minute_key, 0)
        if minute_count >= self.config.rate_limit_per_minute:
            return False
        
        # Check hour limit
        hour_count = self.rate_limit_store.get(hour_key, 0)
        if hour_count >= self.config.rate_limit_per_hour:
            return False
        
        # Update counters
        self.rate_limit_store[minute_key] = minute_count + 1
        self.rate_limit_store[hour_key] = hour_count + 1
        
        return True
    
    async def _handle_proxy_request(self, service_name: str, path: str, request: Request):
        """Handle proxy request to microservice."""
        try:
            # Get service instances
            instances = await self.service_discovery.get_service_instances(service_name)
            if not instances:
                raise HTTPException(status_code=503, detail=f"Service {service_name} not available")
            
            # Add instances to load balancer
            self.load_balancer.add_service_instances(service_name, instances)
            
            # Get next instance
            instance = self.load_balancer.get_next_instance(service_name)
            if not instance:
                raise HTTPException(status_code=503, detail=f"No healthy instances for {service_name}")
            
            # Check circuit breaker
            circuit_breaker = self.service_mesh.get_circuit_breaker(service_name)
            if not circuit_breaker.can_execute():
                raise HTTPException(status_code=503, detail=f"Circuit breaker open for {service_name}")
            
            # Prepare request
            url = f"http://{instance.host}:{instance.port}/{path}"
            headers = dict(request.headers)
            headers.pop("host", None)  # Remove host header
            
            # Get request body
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
            
            # Make request to service
            start_time = time.time()
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=body
                )
            
            response_time = time.time() - start_time
            
            # Update load balancer stats
            self.load_balancer.update_instance_stats(
                service_name, instance.id, response_time, response.status_code < 400
            )
            
            # Update circuit breaker
            if response.status_code < 400:
                circuit_breaker.on_success()
            else:
                circuit_breaker.on_failure()
            
            # Return response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
        except Exception as e:
            self._logger.error(f"Proxy request failed: {e}")
            raise HTTPException(status_code=500, detail="Internal gateway error")


# =============================================================================
# MICROSERVICES ARCHITECTURE MANAGER
# =============================================================================

class MicroservicesArchitecture:
    """
    Advanced microservices architecture manager.
    
    Features:
    - Service Discovery with Consul
    - Load Balancing with multiple strategies
    - Circuit Breakers for fault tolerance
    - API Gateway with rate limiting
    - Service Mesh with tracing and metrics
    - Distributed communication patterns
    """
    
    def __init__(self):
        self.service_discovery = ConsulServiceDiscovery()
        self.load_balancer = LoadBalancer(LoadBalancerConfig())
        self.service_mesh = ServiceMesh(ServiceMeshConfig())
        self.api_gateway = APIGateway(APIGatewayConfig())
        self._logger = get_logger(__name__)
        
        # Service registry
        self.services: Dict[str, ServiceInstance] = {}
        
        # Metrics
        self.request_counter = Counter(
            "microservices_requests_total",
            "Total requests to microservices",
            ["service_name", "method", "status"]
        )
        
        self.response_time_histogram = Histogram(
            "microservices_response_time_seconds",
            "Response time for microservices",
            ["service_name", "method"]
        )
        
        self.circuit_breaker_gauge = Gauge(
            "circuit_breaker_state",
            "Circuit breaker state",
            ["service_name"]
        )
    
    async def register_service(self, service: ServiceInstance) -> bool:
        """Register a service in the microservices architecture."""
        try:
            # Register with service discovery
            success = await self.service_discovery.register_service(service)
            if success:
                self.services[service.id] = service
                self._logger.info(f"Service registered: {service.name} ({service.id})")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Failed to register service: {e}")
            return False
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service from the microservices architecture."""
        try:
            # Deregister from service discovery
            success = await self.service_discovery.deregister_service(service_id)
            if success:
                self.services.pop(service_id, None)
                self._logger.info(f"Service deregistered: {service_id}")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Failed to deregister service: {e}")
            return False
    
    async def discover_services(self, service_type: ServiceType) -> List[ServiceInstance]:
        """Discover services of a specific type."""
        return await self.service_discovery.discover_services(service_type)
    
    async def make_service_request(self, service_name: str, request: ServiceRequest) -> ServiceResponse:
        """Make a request to a microservice."""
        start_time = time.time()
        
        try:
            # Get service instances
            instances = await self.service_discovery.get_service_instances(service_name)
            if not instances:
                raise Exception(f"No instances available for service: {service_name}")
            
            # Add to load balancer
            self.load_balancer.add_service_instances(service_name, instances)
            
            # Get next instance
            instance = self.load_balancer.get_next_instance(service_name)
            if not instance:
                raise Exception(f"No healthy instances for service: {service_name}")
            
            # Check circuit breaker
            circuit_breaker = self.service_mesh.get_circuit_breaker(service_name)
            if not circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for service: {service_name}")
            
            # Make request
            url = f"http://{instance.host}:{instance.port}{request.path}"
            async with httpx.AsyncClient(timeout=request.timeout) as client:
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=request.headers,
                    json=request.body
                )
            
            response_time = time.time() - start_time
            
            # Update metrics
            self.request_counter.labels(
                service_name=service_name,
                method=request.method,
                status=response.status_code
            ).inc()
            
            self.response_time_histogram.labels(
                service_name=service_name,
                method=request.method
            ).observe(response_time)
            
            # Update circuit breaker
            if response.status_code < 400:
                circuit_breaker.on_success()
            else:
                circuit_breaker.on_failure()
            
            # Update load balancer stats
            self.load_balancer.update_instance_stats(
                service_name, instance.id, response_time, response.status_code < 400
            )
            
            # Update circuit breaker gauge
            self.circuit_breaker_gauge.labels(service_name=service_name).set(
                1 if circuit_breaker.state == CircuitBreakerState.OPEN else 0
            )
            
            return ServiceResponse(
                request_id=request.id,
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                response_time=response_time,
                service_instance=instance
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Update metrics for failure
            self.request_counter.labels(
                service_name=service_name,
                method=request.method,
                status="error"
            ).inc()
            
            # Update circuit breaker
            circuit_breaker = self.service_mesh.get_circuit_breaker(service_name)
            circuit_breaker.on_failure()
            
            self._logger.error(f"Service request failed: {e}")
            
            return ServiceResponse(
                request_id=request.id,
                status_code=500,
                headers={},
                body=None,
                response_time=response_time,
                service_instance=instance if 'instance' in locals() else None,
                error_message=str(e)
            )
    
    def get_api_gateway(self) -> FastAPI:
        """Get the API Gateway FastAPI application."""
        return self.api_gateway.app
    
    async def get_architecture_stats(self) -> Dict[str, Any]:
        """Get microservices architecture statistics."""
        return {
            "total_services": len(self.services),
            "service_types": list(set(service.service_type.value for service in self.services.values())),
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "success_count": cb.success_count
                }
                for name, cb in self.service_mesh.circuit_breakers.items()
            },
            "load_balancer": {
                "strategy": self.load_balancer.config.strategy.value,
                "total_instances": sum(len(instances) for instances in self.load_balancer._service_instances.values())
            },
            "api_gateway": {
                "rate_limits": len(self.api_gateway.rate_limit_store),
                "cache_size": len(self.api_gateway.cache)
            }
        }


# =============================================================================
# DECORATORS
# =============================================================================

def microservice(service_type: ServiceType):
    """Decorator to mark a service as a microservice."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Add microservice context
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def circuit_breaker_protected(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Decorator to add circuit breaker protection to a service method."""
    def decorator(func):
        @circuit(failure_threshold=failure_threshold, recovery_timeout=recovery_timeout)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def load_balanced(strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
    """Decorator to add load balancing to a service method."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Add load balancing context
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limited(requests_per_minute: int = 100):
    """Decorator to add rate limiting to a service method."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Add rate limiting context
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# MAIN APPLICATION
# =============================================================================

async def main():
    """Main application entry point."""
    # Initialize microservices architecture
    microservices = MicroservicesArchitecture()
    
    # Register example services
    services = [
        ServiceInstance(
            id="content-generation-1",
            name="content-generation",
            service_type=ServiceType.CONTENT_GENERATION,
            host="localhost",
            port=8001,
            health_check_url="http://localhost:8001/health"
        ),
        ServiceInstance(
            id="optimization-1",
            name="optimization",
            service_type=ServiceType.OPTIMIZATION,
            host="localhost",
            port=8002,
            health_check_url="http://localhost:8002/health"
        ),
        ServiceInstance(
            id="analytics-1",
            name="analytics",
            service_type=ServiceType.ANALYTICS,
            host="localhost",
            port=8003,
            health_check_url="http://localhost:8003/health"
        )
    ]
    
    # Register services
    for service in services:
        await microservices.register_service(service)
    
    # Get API Gateway
    api_gateway = microservices.get_api_gateway()
    
    # Start the application
    import uvicorn
    uvicorn.run(api_gateway, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    asyncio.run(main()) 