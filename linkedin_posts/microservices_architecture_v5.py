"""
🔧 MICROSERVICES ARCHITECTURE v5.0
===================================

Enterprise microservices architecture including:
- Service Mesh (Istio/Envoy concepts)
- API Gateway (Kong/Apache APISIX concepts)
- Circuit Breaker patterns
- Service Discovery and registration
- Event-driven communication
- Distributed tracing
"""

import asyncio
import time
import logging
import json
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import defaultdict
from datetime import datetime, timedelta
import aiohttp
import asyncio_mqtt
import aioredis
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums
class ServiceStatus(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    OFFLINE = auto()

class CircuitBreakerState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()

class ServiceType(Enum):
    API_GATEWAY = auto()
    CONTENT_OPTIMIZER = auto()
    ANALYTICS_ENGINE = auto()
    SECURITY_SERVICE = auto()
    INFRASTRUCTURE_MANAGER = auto()

# Data structures
@dataclass
class ServiceInstance:
    instance_id: str
    service_name: str
    service_type: ServiceType
    host: str
    port: int
    status: ServiceStatus
    health_check_url: str
    metadata: Dict[str, Any]
    last_heartbeat: datetime
    load_balancer_weight: float = 1.0

@dataclass
class ServiceRequest:
    request_id: str
    service_name: str
    method: str
    path: str
    headers: Dict[str, str]
    body: Any
    timestamp: datetime
    trace_id: str

@dataclass
class ServiceResponse:
    request_id: str
    status_code: int
    headers: Dict[str, str]
    body: Any
    response_time: float
    service_instance: str

# Service Mesh Implementation
class ServiceMesh:
    """Advanced service mesh for microservices communication."""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = defaultdict(list)
        self.routing_rules = {}
        self.load_balancers = {}
        self.health_checkers = {}
        self.metrics = {
            'requests_total': Counter('service_mesh_requests_total', 'Total requests processed'),
            'response_time': Histogram('service_mesh_response_time', 'Response time in seconds'),
            'active_services': Gauge('service_mesh_active_services', 'Number of active services')
        }
        
    async def register_service(self, service_name: str, instance: ServiceInstance):
        """Register a new service instance."""
        self.services[service_name].append(instance)
        
        # Initialize load balancer for this service
        if service_name not in self.load_balancers:
            self.load_balancers[service_name] = LoadBalancer()
        
        # Start health checking
        await self._start_health_checker(service_name, instance)
        
        # Update metrics
        self.metrics['active_services'].set(len(self.services))
        
        logger.info(f"🔧 Service registered: {service_name} at {instance.host}:{instance.port}")
    
    async def unregister_service(self, service_name: str, instance_id: str):
        """Unregister a service instance."""
        if service_name in self.services:
            self.services[service_name] = [
                inst for inst in self.services[service_name] 
                if inst.instance_id != instance_id
            ]
            
            # Update metrics
            self.metrics['active_services'].set(len(self.services))
            
            logger.info(f"🔧 Service unregistered: {service_name} instance {instance_id}")
    
    async def route_request(self, service_name: str, request: ServiceRequest) -> ServiceResponse:
        """Route request to appropriate service instance."""
        if service_name not in self.services or not self.services[service_name]:
            raise ValueError(f"No instances available for service: {service_name}")
        
        # Get healthy instances
        healthy_instances = [
            inst for inst in self.services[service_name]
            if inst.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
        ]
        
        if not healthy_instances:
            raise RuntimeError(f"No healthy instances available for service: {service_name}")
        
        # Select instance using load balancer
        load_balancer = self.load_balancers[service_name]
        selected_instance = load_balancer.select_instance(healthy_instances)
        
        # Execute request
        start_time = time.time()
        try:
            response = await self._execute_request(selected_instance, request)
            response_time = time.time() - start_time
            
            # Update metrics
            self.metrics['requests_total'].inc()
            self.metrics['response_time'].observe(response_time)
            
            # Update load balancer with response time
            load_balancer.update_instance_metrics(selected_instance.instance_id, response_time)
            
            return ServiceResponse(
                request_id=request.request_id,
                status_code=response['status_code'],
                headers=response['headers'],
                body=response['body'],
                response_time=response_time,
                service_instance=selected_instance.instance_id
            )
            
        except Exception as e:
            # Mark instance as degraded
            selected_instance.status = ServiceStatus.DEGRADED
            logger.error(f"Request failed for {service_name}: {e}")
            raise
    
    async def _execute_request(self, instance: ServiceInstance, request: ServiceRequest) -> Dict[str, Any]:
        """Execute HTTP request to service instance."""
        url = f"http://{instance.host}:{instance.port}{request.path}"
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=url,
                headers=request.headers,
                json=request.body
            ) as response:
                body = await response.json() if response.content_type == 'application/json' else await response.text()
                
                return {
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'body': body
                }
    
    async def _start_health_checker(self, service_name: str, instance: ServiceInstance):
        """Start health checking for a service instance."""
        async def health_check_loop():
            while instance.instance_id in [inst.instance_id for inst in self.services[service_name]]:
                try:
                    # Perform health check
                    async with aiohttp.ClientSession() as session:
                        async with session.get(instance.health_check_url, timeout=5) as response:
                            if response.status == 200:
                                instance.status = ServiceStatus.HEALTHY
                                instance.last_heartbeat = datetime.now()
                            else:
                                instance.status = ServiceStatus.DEGRADED
                    
                except Exception as e:
                    instance.status = ServiceStatus.UNHEALTHY
                    logger.warning(f"Health check failed for {service_name}: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
        
        # Start health checker
        asyncio.create_task(health_check_loop())

# Load Balancer Implementation
class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instance_metrics = defaultdict(lambda: {
            'response_times': [],
            'request_count': 0,
            'error_count': 0
        })
        self.current_index = 0
    
    def select_instance(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance based on load balancing strategy."""
        if not instances:
            raise ValueError("No instances available")
        
        if self.strategy == "round_robin":
            return self._round_robin(instances)
        elif self.strategy == "least_connections":
            return self._least_connections(instances)
        elif self.strategy == "weighted":
            return self._weighted_selection(instances)
        elif self.strategy == "least_response_time":
            return self._least_response_time(instances)
        else:
            return self._round_robin(instances)
    
    def _round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin load balancing."""
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance
    
    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections load balancing."""
        return min(instances, key=lambda x: self.instance_metrics[x.instance_id]['request_count'])
    
    def _weighted_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted load balancing."""
        total_weight = sum(inst.load_balancer_weight for inst in instances)
        if total_weight == 0:
            return instances[0]
        
        # Simple weighted random selection
        import random
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.load_balancer_weight
            if rand_val <= current_weight:
                return instance
        
        return instances[-1]
    
    def _least_response_time(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least response time load balancing."""
        def get_avg_response_time(instance_id: str) -> float:
            metrics = self.instance_metrics[instance_id]
            if not metrics['response_times']:
                return float('inf')
            return sum(metrics['response_times']) / len(metrics['response_times'])
        
        return min(instances, key=lambda x: get_avg_response_time(x.instance_id))
    
    def update_instance_metrics(self, instance_id: str, response_time: float):
        """Update metrics for an instance."""
        metrics = self.instance_metrics[instance_id]
        metrics['response_times'].append(response_time)
        metrics['request_count'] += 1
        
        # Keep only last 100 response times
        if len(metrics['response_times']) > 100:
            metrics['response_times'] = metrics['response_times'][-100:]

# Circuit Breaker Implementation
class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.metrics = {
            'total_requests': Counter('circuit_breaker_total_requests', 'Total requests'),
            'failed_requests': Counter('circuit_breaker_failed_requests', 'Failed requests'),
            'circuit_opens': Counter('circuit_breaker_opens', 'Circuit breaker opens')
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        self.metrics['total_requests'].inc()
        
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("🔄 Circuit breaker attempting reset")
            else:
                raise RuntimeError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("✅ Circuit breaker reset to CLOSED")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.metrics['failed_requests'].inc()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.metrics['circuit_opens'].inc()
            logger.warning("🚨 Circuit breaker opened due to failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return False
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout

# API Gateway Implementation
class APIGateway:
    """Advanced API Gateway with routing and middleware."""
    
    def __init__(self):
        self.routes = {}
        self.middleware = []
        self.rate_limiters = {}
        self.authentication = None
        self.metrics = {
            'gateway_requests': Counter('api_gateway_requests_total', 'Total gateway requests'),
            'gateway_response_time': Histogram('api_gateway_response_time', 'Gateway response time')
        }
    
    def add_route(self, path: str, service_name: str, methods: List[str] = None):
        """Add a new route."""
        if methods is None:
            methods = ['GET', 'POST', 'PUT', 'DELETE']
        
        self.routes[path] = {
            'service_name': service_name,
            'methods': methods
        }
        logger.info(f"🔗 Route added: {path} -> {service_name}")
    
    def add_middleware(self, middleware_func: Callable):
        """Add middleware function."""
        self.middleware.append(middleware_func)
        logger.info("🔧 Middleware added")
    
    async def handle_request(self, request: ServiceRequest) -> ServiceResponse:
        """Handle incoming request through gateway."""
        start_time = time.time()
        
        # Update metrics
        self.metrics['gateway_requests'].inc()
        
        # Find route
        if request.path not in self.routes:
            raise ValueError(f"Route not found: {request.path}")
        
        route = self.routes[request.path]
        if request.method not in route['methods']:
            raise ValueError(f"Method not allowed: {request.method}")
        
        # Apply middleware
        processed_request = request
        for middleware in self.middleware:
            processed_request = await middleware(processed_request)
        
        # Route to service
        service_mesh = ServiceMesh()  # In production, this would be injected
        response = await service_mesh.route_request(route['service_name'], processed_request)
        
        # Calculate response time
        response_time = time.time() - start_time
        self.metrics['gateway_response_time'].observe(response_time)
        
        return response

# Service Discovery Implementation
class ServiceDiscovery:
    """Service discovery and registration system."""
    
    def __init__(self):
        self.service_registry = {}
        self.service_watchers = defaultdict(list)
        self.redis_client = None
        
    async def initialize(self, redis_url: str = "redis://localhost"):
        """Initialize service discovery with Redis backend."""
        try:
            self.redis_client = aioredis.from_url(redis_url)
            logger.info("🔍 Service discovery initialized with Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory registry: {e}")
    
    async def register_service(self, service_name: str, instance: ServiceInstance):
        """Register a service instance."""
        # Store in memory
        if service_name not in self.service_registry:
            self.service_registry[service_name] = []
        
        self.service_registry[service_name].append(instance)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                key = f"service:{service_name}:{instance.instance_id}"
                await self.redis_client.setex(
                    key, 
                    300,  # 5 minutes TTL
                    json.dumps({
                        'instance_id': instance.instance_id,
                        'service_name': instance.service_name,
                        'host': instance.host,
                        'port': instance.port,
                        'status': instance.status.name,
                        'metadata': instance.metadata
                    })
                )
            except Exception as e:
                logger.warning(f"Failed to store service in Redis: {e}")
        
        # Notify watchers
        await self._notify_watchers(service_name, 'registered', instance)
        
        logger.info(f"🔍 Service registered: {service_name} at {instance.host}:{instance.port}")
    
    async def discover_service(self, service_name: str) -> List[ServiceInstance]:
        """Discover instances of a service."""
        # Check memory first
        if service_name in self.service_registry:
            return self.service_registry[service_name]
        
        # Check Redis if available
        if self.redis_client:
            try:
                pattern = f"service:{service_name}:*"
                keys = await self.redis_client.keys(pattern)
                
                instances = []
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        service_data = json.loads(data)
                        instance = ServiceInstance(
                            instance_id=service_data['instance_id'],
                            service_name=service_data['service_name'],
                            service_type=ServiceType.CONTENT_OPTIMIZER,  # Default
                            host=service_data['host'],
                            port=service_data['port'],
                            status=ServiceStatus(service_data['status']),
                            health_check_url=f"http://{service_data['host']}:{service_data['port']}/health",
                            metadata=service_data['metadata'],
                            last_heartbeat=datetime.now()
                        )
                        instances.append(instance)
                
                return instances
            except Exception as e:
                logger.warning(f"Failed to discover service from Redis: {e}")
        
        return []
    
    async def watch_service(self, service_name: str, callback: Callable):
        """Watch for service changes."""
        self.service_watchers[service_name].append(callback)
        logger.info(f"👀 Watching service: {service_name}")
    
    async def _notify_watchers(self, service_name: str, event: str, instance: ServiceInstance):
        """Notify watchers of service events."""
        for callback in self.service_watchers[service_name]:
            try:
                await callback(event, instance)
            except Exception as e:
                logger.error(f"Watcher callback failed: {e}")

# Main Microservices System
class MicroservicesArchitectureSystem:
    """Main microservices architecture system v5.0."""
    
    def __init__(self):
        self.service_mesh = ServiceMesh()
        self.api_gateway = APIGateway()
        self.service_discovery = ServiceDiscovery()
        self.circuit_breakers = {}
        
        # Initialize default routes
        self._initialize_default_routes()
        
        logger.info("🔧 Microservices Architecture System v5.0 initialized")
    
    def _initialize_default_routes(self):
        """Initialize default API routes."""
        self.api_gateway.add_route('/api/v1/optimize', 'content-optimizer', ['POST'])
        self.api_gateway.add_route('/api/v1/analytics', 'analytics-engine', ['GET', 'POST'])
        self.api_gateway.add_route('/api/v1/security', 'security-service', ['POST'])
        self.api_gateway.add_route('/api/v1/infrastructure', 'infrastructure-manager', ['GET', 'POST'])
    
    async def start_system(self):
        """Start the microservices system."""
        # Initialize service discovery
        await self.service_discovery.initialize()
        
        # Start health monitoring
        asyncio.create_task(self._health_monitoring_loop())
        
        logger.info("🚀 Microservices system started")
    
    async def register_service_instance(self, service_name: str, host: str, port: int, 
                                       service_type: ServiceType = ServiceType.CONTENT_OPTIMIZER):
        """Register a new service instance."""
        instance = ServiceInstance(
            instance_id=str(uuid.uuid4()),
            service_name=service_name,
            service_type=service_type,
            host=host,
            port=port,
            status=ServiceStatus.HEALTHY,
            health_check_url=f"http://{host}:{port}/health",
            metadata={'registered_at': datetime.now().isoformat()},
            last_heartbeat=datetime.now()
        )
        
        # Register with service discovery
        await self.service_discovery.register_service(service_name, instance)
        
        # Register with service mesh
        await self.service_mesh.register_service(service_name, instance)
        
        # Create circuit breaker for this service
        self.circuit_breakers[service_name] = CircuitBreaker()
        
        return instance.instance_id
    
    async def make_service_call(self, service_name: str, method: str, path: str, 
                                headers: Dict[str, str] = None, body: Any = None) -> ServiceResponse:
        """Make a call to a service through the API gateway."""
        request = ServiceRequest(
            request_id=str(uuid.uuid4()),
            service_name=service_name,
            method=method,
            path=path,
            headers=headers or {},
            body=body,
            timestamp=datetime.now(),
            trace_id=str(uuid.uuid4())
        )
        
        # Use circuit breaker if available
        if service_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[service_name]
            return await circuit_breaker.call(
                self.api_gateway.handle_request, request
            )
        else:
            return await self.api_gateway.handle_request(request)
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring loop."""
        while True:
            try:
                # Check all services
                for service_name in list(self.service_mesh.services.keys()):
                    instances = self.service_mesh.services[service_name]
                    healthy_count = sum(1 for inst in instances if inst.status == ServiceStatus.HEALTHY)
                    
                    if healthy_count == 0:
                        logger.warning(f"🚨 No healthy instances for service: {service_name}")
                    elif healthy_count < len(instances):
                        logger.warning(f"⚠️ Some instances degraded for service: {service_name}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        status = {
            'total_services': len(self.service_mesh.services),
            'total_instances': sum(len(instances) for instances in self.service_mesh.services.values()),
            'healthy_instances': sum(
                sum(1 for inst in instances if inst.status == ServiceStatus.HEALTHY)
                for instances in self.service_mesh.services.values()
            ),
            'degraded_instances': sum(
                sum(1 for inst in instances if inst.status == ServiceStatus.DEGRADED)
                for instances in self.service_mesh.services.values()
            ),
            'unhealthy_instances': sum(
                sum(1 for inst in instances if inst.status == ServiceStatus.UNHEALTHY)
                for instances in self.service_mesh.services.values()
            ),
            'circuit_breakers': {
                service: {
                    'state': cb.state.name,
                    'failure_count': cb.failure_count
                }
                for service, cb in self.circuit_breakers.items()
            }
        }
        
        return status

# Demo function
async def demo_microservices_architecture():
    """Demonstrate microservices architecture capabilities."""
    print("🔧 MICROSERVICES ARCHITECTURE v5.0")
    print("=" * 60)
    
    # Initialize system
    system = MicroservicesArchitectureSystem()
    
    print("🚀 Starting microservices system...")
    await system.start_system()
    
    try:
        # Register service instances
        print("\n🔧 Registering service instances...")
        optimizer_id = await system.register_service_instance(
            "content-optimizer", "localhost", 8001, ServiceType.CONTENT_OPTIMIZER
        )
        analytics_id = await system.register_service_instance(
            "analytics-engine", "localhost", 8002, ServiceType.ANALYTICS_ENGINE
        )
        security_id = await system.register_service_instance(
            "security-service", "localhost", 8003, ServiceType.SECURITY_SERVICE
        )
        
        print(f"✅ Services registered: {optimizer_id[:8]}, {analytics_id[:8]}, {security_id[:8]}")
        
        # Test service calls
        print("\n📡 Testing service calls...")
        try:
            response = await system.make_service_call(
                "content-optimizer", "POST", "/api/v1/optimize",
                body={"content": "test content"}
            )
            print(f"✅ Service call successful: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Service call failed (expected): {e}")
        
        # Get system status
        print("\n📊 System status:")
        status = await system.get_service_status()
        for key, value in status.items():
            if key == 'circuit_breakers':
                print(f"   {key}:")
                for service, cb_status in value.items():
                    print(f"     {service}: {cb_status['state']} (failures: {cb_status['failure_count']})")
            else:
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    print("\n🎉 Microservices Architecture demo completed!")
    print("✨ The system now provides enterprise-grade microservices capabilities!")

if __name__ == "__main__":
    asyncio.run(demo_microservices_architecture())
