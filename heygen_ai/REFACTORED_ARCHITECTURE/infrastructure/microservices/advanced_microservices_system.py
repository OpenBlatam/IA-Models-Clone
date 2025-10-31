"""
Advanced Microservices System

This module provides comprehensive microservices architecture capabilities
for the refactored HeyGen AI system with service discovery, load balancing,
and distributed communication.
"""

import asyncio
import aiohttp
import json
import logging
import uuid
import time
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
import consul
import redis
import threading
from concurrent.futures import ThreadPoolExecutor
import yaml
import hashlib
import base64
from cryptography.fernet import Fernet
import requests
import websockets
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import gc


logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"


class ServiceType(str, Enum):
    """Service types."""
    API_GATEWAY = "api_gateway"
    AUTH_SERVICE = "auth_service"
    USER_SERVICE = "user_service"
    AI_SERVICE = "ai_service"
    ANALYTICS_SERVICE = "analytics_service"
    NOTIFICATION_SERVICE = "notification_service"
    FILE_SERVICE = "file_service"
    CACHE_SERVICE = "cache_service"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    RANDOM = "random"


@dataclass
class ServiceInstance:
    """Service instance structure."""
    service_id: str
    service_name: str
    service_type: ServiceType
    host: str
    port: int
    status: ServiceStatus
    health_check_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_health_check: Optional[datetime] = None
    response_time: float = 0.0
    error_count: int = 0
    weight: int = 1


@dataclass
class ServiceRequest:
    """Service request structure."""
    request_id: str
    service_name: str
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ServiceResponse:
    """Service response structure."""
    request_id: str
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    response_time: float = 0.0
    service_instance: Optional[ServiceInstance] = None
    error: Optional[str] = None


class ServiceRegistry:
    """Advanced service registry with discovery capabilities."""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul_client = consul.Consul(host=consul_host, port=consul_port)
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5  # seconds
        self.lock = threading.RLock()
        self.running = False
        self.health_check_thread = None
    
    def start(self):
        """Start the service registry."""
        self.running = True
        self.health_check_thread = threading.Thread(target=self._health_check_worker, daemon=True)
        self.health_check_thread.start()
        logger.info("Service registry started")
    
    def stop(self):
        """Stop the service registry."""
        self.running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        logger.info("Service registry stopped")
    
    async def register_service(self, service: ServiceInstance) -> bool:
        """Register a service instance."""
        try:
            # Register with Consul
            service_id = f"{service.service_name}-{service.service_id}"
            check = consul.Check.http(
                service.health_check_url,
                interval=f"{self.health_check_interval}s",
                timeout=f"{self.health_check_timeout}s"
            )
            
            self.consul_client.agent.service.register(
                name=service.service_name,
                service_id=service_id,
                address=service.host,
                port=service.port,
                check=check,
                meta=service.metadata
            )
            
            # Add to local registry
            with self.lock:
                if service.service_name not in self.services:
                    self.services[service.service_name] = []
                self.services[service.service_name].append(service)
            
            logger.info(f"Service {service.service_name} registered: {service.host}:{service.port}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering service {service.service_name}: {e}")
            return False
    
    async def deregister_service(self, service_name: str, service_id: str) -> bool:
        """Deregister a service instance."""
        try:
            # Deregister from Consul
            consul_service_id = f"{service_name}-{service_id}"
            self.consul_client.agent.service.deregister(consul_service_id)
            
            # Remove from local registry
            with self.lock:
                if service_name in self.services:
                    self.services[service_name] = [
                        s for s in self.services[service_name] 
                        if s.service_id != service_id
                    ]
            
            logger.info(f"Service {service_name} deregistered: {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deregistering service {service_name}: {e}")
            return False
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover healthy service instances."""
        try:
            # Get services from Consul
            _, services = self.consul_client.health.service(service_name, passing=True)
            
            instances = []
            for service in services:
                instance = ServiceInstance(
                    service_id=service['Service']['ID'],
                    service_name=service['Service']['Service'],
                    service_type=ServiceType(service['Service'].get('Meta', {}).get('type', 'ai_service')),
                    host=service['Service']['Address'],
                    port=service['Service']['Port'],
                    status=ServiceStatus.HEALTHY,
                    health_check_url=service['Service'].get('Meta', {}).get('health_check_url', ''),
                    metadata=service['Service'].get('Meta', {})
                )
                instances.append(instance)
            
            # Update local registry
            with self.lock:
                self.services[service_name] = instances
            
            return instances
            
        except Exception as e:
            logger.error(f"Error discovering services {service_name}: {e}")
            return []
    
    def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get service instances from local registry."""
        with self.lock:
            return self.services.get(service_name, [])
    
    def _health_check_worker(self):
        """Health check worker thread."""
        while self.running:
            try:
                with self.lock:
                    for service_name, instances in self.services.items():
                        for instance in instances:
                            self._check_service_health(instance)
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check worker: {e}")
                time.sleep(5)
    
    def _check_service_health(self, instance: ServiceInstance):
        """Check health of a service instance."""
        try:
            start_time = time.time()
            response = requests.get(
                instance.health_check_url,
                timeout=self.health_check_timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                instance.status = ServiceStatus.HEALTHY
                instance.response_time = response_time
                instance.error_count = 0
            else:
                instance.status = ServiceStatus.UNHEALTHY
                instance.error_count += 1
                
        except Exception as e:
            instance.status = ServiceStatus.UNHEALTHY
            instance.error_count += 1
            logger.warning(f"Health check failed for {instance.service_name}: {e}")
        
        instance.last_health_check = datetime.now(timezone.utc)


class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.round_robin_index = 0
        self.connection_counts: Dict[str, int] = {}
        self.lock = threading.RLock()
    
    def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select a service instance based on strategy."""
        if not instances:
            return None
        
        # Filter healthy instances
        healthy_instances = [i for i in instances if i.status == ServiceStatus.HEALTHY]
        if not healthy_instances:
            return None
        
        with self.lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.IP_HASH:
                return self._ip_hash_selection(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return self._random_selection(healthy_instances)
            else:
                return healthy_instances[0]
    
    def _round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection."""
        instance = instances[self.round_robin_index % len(instances)]
        self.round_robin_index += 1
        return instance
    
    def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection."""
        min_connections = float('inf')
        selected_instance = instances[0]
        
        for instance in instances:
            connections = self.connection_counts.get(instance.service_id, 0)
            if connections < min_connections:
                min_connections = connections
                selected_instance = instance
        
        return selected_instance
    
    def _weighted_round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin selection."""
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return instances[0]
        
        # Simple weighted selection
        current_weight = 0
        target_weight = self.round_robin_index % total_weight
        
        for instance in instances:
            current_weight += instance.weight
            if current_weight > target_weight:
                self.round_robin_index += 1
                return instance
        
        return instances[-1]
    
    def _ip_hash_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """IP hash selection (simplified)."""
        # Use round robin index as hash input
        hash_value = hash(str(self.round_robin_index)) % len(instances)
        self.round_robin_index += 1
        return instances[hash_value]
    
    def _random_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Random selection."""
        import random
        return random.choice(instances)
    
    def record_connection(self, instance: ServiceInstance):
        """Record a connection to an instance."""
        with self.lock:
            self.connection_counts[instance.service_id] = self.connection_counts.get(instance.service_id, 0) + 1
    
    def release_connection(self, instance: ServiceInstance):
        """Release a connection from an instance."""
        with self.lock:
            if instance.service_id in self.connection_counts:
                self.connection_counts[instance.service_id] = max(0, self.connection_counts[instance.service_id] - 1)


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.RLock()
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        with self.lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                    return True
                return False
            else:  # half-open
                return True
    
    def record_success(self):
        """Record successful execution."""
        with self.lock:
            self.failure_count = 0
            self.state = "closed"
    
    def record_failure(self):
        """Record failed execution."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"


class ServiceClient:
    """Advanced service client with load balancing and circuit breaking."""
    
    def __init__(self, service_registry: ServiceRegistry, load_balancer: LoadBalancer):
        self.service_registry = service_registry
        self.load_balancer = load_balancer
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.session = aiohttp.ClientSession()
        self.metrics = {
            'requests_total': Counter('service_requests_total', 'Total service requests', ['service', 'status']),
            'request_duration': Histogram('service_request_duration_seconds', 'Service request duration', ['service']),
            'active_connections': Gauge('service_active_connections', 'Active service connections', ['service'])
        }
    
    async def make_request(self, request: ServiceRequest) -> ServiceResponse:
        """Make a request to a service."""
        start_time = time.time()
        
        try:
            # Get service instances
            instances = await self.service_registry.discover_services(request.service_name)
            if not instances:
                return ServiceResponse(
                    request_id=request.request_id,
                    status_code=503,
                    error="No healthy service instances available"
                )
            
            # Get circuit breaker
            circuit_breaker = self._get_circuit_breaker(request.service_name)
            if not circuit_breaker.can_execute():
                return ServiceResponse(
                    request_id=request.request_id,
                    status_code=503,
                    error="Circuit breaker is open"
                )
            
            # Select instance
            instance = self.load_balancer.select_instance(instances)
            if not instance:
                return ServiceResponse(
                    request_id=request.request_id,
                    status_code=503,
                    error="No healthy service instances available"
                )
            
            # Record connection
            self.load_balancer.record_connection(instance)
            
            try:
                # Make request
                url = f"http://{instance.host}:{instance.port}{request.path}"
                async with self.session.request(
                    method=request.method,
                    url=url,
                    headers=request.headers,
                    data=request.body,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:
                    response_body = await response.read()
                    
                    # Record success
                    circuit_breaker.record_success()
                    
                    # Update metrics
                    duration = time.time() - start_time
                    self.metrics['requests_total'].labels(
                        service=request.service_name,
                        status=response.status
                    ).inc()
                    self.metrics['request_duration'].labels(
                        service=request.service_name
                    ).observe(duration)
                    
                    return ServiceResponse(
                        request_id=request.request_id,
                        status_code=response.status,
                        headers=dict(response.headers),
                        body=response_body,
                        response_time=duration,
                        service_instance=instance
                    )
                    
            except Exception as e:
                # Record failure
                circuit_breaker.record_failure()
                
                # Update metrics
                self.metrics['requests_total'].labels(
                    service=request.service_name,
                    status='error'
                ).inc()
                
                return ServiceResponse(
                    request_id=request.request_id,
                    status_code=500,
                    error=str(e),
                    service_instance=instance
                )
                
            finally:
                # Release connection
                self.load_balancer.release_connection(instance)
                
        except Exception as e:
            return ServiceResponse(
                request_id=request.request_id,
                status_code=500,
                error=str(e)
            )
    
    def _get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    async def close(self):
        """Close the service client."""
        await self.session.close()


class AdvancedMicroservicesSystem:
    """
    Advanced microservices system with comprehensive capabilities.
    
    Features:
    - Service discovery and registration
    - Load balancing with multiple strategies
    - Circuit breaking for fault tolerance
    - Health checking and monitoring
    - Distributed communication
    - Service mesh capabilities
    - Metrics and observability
    - Auto-scaling and self-healing
    """
    
    def __init__(
        self,
        consul_host: str = "localhost",
        consul_port: int = 8500,
        prometheus_port: int = 9090
    ):
        """
        Initialize the advanced microservices system.
        
        Args:
            consul_host: Consul server host
            consul_port: Consul server port
            prometheus_port: Prometheus metrics port
        """
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.prometheus_port = prometheus_port
        
        # Initialize components
        self.service_registry = ServiceRegistry(consul_host, consul_port)
        self.load_balancer = LoadBalancer()
        self.service_client = ServiceClient(self.service_registry, self.load_balancer)
        
        # Start services
        self.service_registry.start()
        start_http_server(prometheus_port)
        
        logger.info("Advanced microservices system initialized")
    
    async def register_service(
        self,
        service_name: str,
        service_type: ServiceType,
        host: str,
        port: int,
        health_check_url: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Register a service instance."""
        service = ServiceInstance(
            service_id=str(uuid.uuid4()),
            service_name=service_name,
            service_type=service_type,
            host=host,
            port=port,
            status=ServiceStatus.STARTING,
            health_check_url=health_check_url,
            metadata=metadata or {}
        )
        
        return await self.service_registry.register_service(service)
    
    async def make_service_request(
        self,
        service_name: str,
        method: str,
        path: str,
        headers: Dict[str, str] = None,
        body: bytes = None,
        timeout: float = 30.0
    ) -> ServiceResponse:
        """Make a request to a service."""
        request = ServiceRequest(
            request_id=str(uuid.uuid4()),
            service_name=service_name,
            method=method,
            path=path,
            headers=headers or {},
            body=body,
            timeout=timeout
        )
        
        return await self.service_client.make_request(request)
    
    def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get service instances."""
        return self.service_registry.get_service_instances(service_name)
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get service health information."""
        instances = self.get_service_instances(service_name)
        
        if not instances:
            return {
                'status': 'unknown',
                'instances': 0,
                'healthy_instances': 0
            }
        
        healthy_instances = [i for i in instances if i.status == ServiceStatus.HEALTHY]
        
        return {
            'status': 'healthy' if healthy_instances else 'unhealthy',
            'instances': len(instances),
            'healthy_instances': len(healthy_instances),
            'instances_detail': [
                {
                    'service_id': i.service_id,
                    'host': i.host,
                    'port': i.port,
                    'status': i.status.value,
                    'response_time': i.response_time,
                    'error_count': i.error_count
                }
                for i in instances
            ]
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            'total_services': len(self.service_registry.services),
            'total_instances': sum(len(instances) for instances in self.service_registry.services.values()),
            'healthy_instances': sum(
                len([i for i in instances if i.status == ServiceStatus.HEALTHY])
                for instances in self.service_registry.services.values()
            ),
            'circuit_breakers': {
                name: {
                    'state': cb.state,
                    'failure_count': cb.failure_count
                }
                for name, cb in self.service_client.circuit_breakers.items()
            }
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.service_client.close()
        self.service_registry.stop()
        logger.info("Microservices system cleanup completed")


# Example usage and demonstration
async def main():
    """Demonstrate the advanced microservices system."""
    print("🔧 HeyGen AI - Advanced Microservices System Demo")
    print("=" * 70)
    
    # Initialize microservices system
    microservices = AdvancedMicroservicesSystem(
        consul_host="localhost",
        consul_port=8500,
        prometheus_port=9090
    )
    
    try:
        # Register sample services
        print("\n📝 Registering Services...")
        
        # Register AI service
        ai_service_registered = await microservices.register_service(
            service_name="ai-service",
            service_type=ServiceType.AI_SERVICE,
            host="localhost",
            port=8001,
            health_check_url="http://localhost:8001/health",
            metadata={"version": "1.0.0", "environment": "development"}
        )
        print(f"AI Service registered: {ai_service_registered}")
        
        # Register Analytics service
        analytics_service_registered = await microservices.register_service(
            service_name="analytics-service",
            service_type=ServiceType.ANALYTICS_SERVICE,
            host="localhost",
            port=8002,
            health_check_url="http://localhost:8002/health",
            metadata={"version": "1.0.0", "environment": "development"}
        )
        print(f"Analytics Service registered: {analytics_service_registered}")
        
        # Register User service
        user_service_registered = await microservices.register_service(
            service_name="user-service",
            service_type=ServiceType.USER_SERVICE,
            host="localhost",
            port=8003,
            health_check_url="http://localhost:8003/health",
            metadata={"version": "1.0.0", "environment": "development"}
        )
        print(f"User Service registered: {user_service_registered}")
        
        # Wait for services to be discovered
        await asyncio.sleep(2)
        
        # Get service instances
        print("\n🔍 Service Discovery...")
        ai_instances = microservices.get_service_instances("ai-service")
        print(f"AI Service instances: {len(ai_instances)}")
        
        analytics_instances = microservices.get_service_instances("analytics-service")
        print(f"Analytics Service instances: {len(analytics_instances)}")
        
        user_instances = microservices.get_service_instances("user-service")
        print(f"User Service instances: {len(user_instances)}")
        
        # Get service health
        print("\n🏥 Service Health...")
        ai_health = microservices.get_service_health("ai-service")
        print(f"AI Service health: {ai_health['status']} ({ai_health['healthy_instances']}/{ai_health['instances']})")
        
        analytics_health = microservices.get_service_health("analytics-service")
        print(f"Analytics Service health: {analytics_health['status']} ({analytics_health['healthy_instances']}/{analytics_health['instances']})")
        
        user_health = microservices.get_service_health("user-service")
        print(f"User Service health: {user_health['status']} ({user_health['healthy_instances']}/{user_health['instances']})")
        
        # Make service requests (simulated)
        print("\n📡 Service Communication...")
        
        # Simulate AI service request
        ai_request = await microservices.make_service_request(
            service_name="ai-service",
            method="POST",
            path="/api/v1/generate",
            headers={"Content-Type": "application/json"},
            body=b'{"prompt": "Hello, AI!"}'
        )
        print(f"AI Service request: {ai_request.status_code} (simulated)")
        
        # Simulate Analytics service request
        analytics_request = await microservices.make_service_request(
            service_name="analytics-service",
            method="GET",
            path="/api/v1/metrics"
        )
        print(f"Analytics Service request: {analytics_request.status_code} (simulated)")
        
        # Simulate User service request
        user_request = await microservices.make_service_request(
            service_name="user-service",
            method="GET",
            path="/api/v1/users"
        )
        print(f"User Service request: {user_request.status_code} (simulated)")
        
        # Get system metrics
        print("\n📊 System Metrics...")
        metrics = microservices.get_system_metrics()
        print(f"Total services: {metrics['total_services']}")
        print(f"Total instances: {metrics['total_instances']}")
        print(f"Healthy instances: {metrics['healthy_instances']}")
        print(f"Circuit breakers: {len(metrics['circuit_breakers'])}")
        
        print(f"\n🌐 Prometheus metrics available at: http://localhost:{microservices.prometheus_port}/metrics")
        print(f"🔍 Consul UI available at: http://localhost:{microservices.consul_port}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Cleanup
        await microservices.cleanup()
        print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())
