"""
Microservices Architecture for Ultimate Opus Clip

Advanced microservices system for scalable, distributed video processing
with service discovery, load balancing, and fault tolerance.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import asyncio
import time
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import aiohttp
from datetime import datetime, timedelta
import hashlib
import random
import socket

logger = structlog.get_logger("microservices_architecture")

class ServiceStatus(Enum):
    """Service status indicators."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"

class ServiceType(Enum):
    """Types of microservices."""
    VIDEO_PROCESSING = "video_processing"
    AI_ANALYSIS = "ai_analysis"
    CLOUD_STORAGE = "cloud_storage"
    USER_MANAGEMENT = "user_management"
    NOTIFICATION = "notification"
    ANALYTICS = "analytics"
    WORKFLOW = "workflow"
    API_GATEWAY = "api_gateway"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    IP_HASH = "ip_hash"

@dataclass
class ServiceInstance:
    """Microservice instance information."""
    instance_id: str
    service_type: ServiceType
    host: str
    port: int
    status: ServiceStatus
    health_check_url: str
    last_health_check: float
    response_time: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    max_connections: int
    weight: int = 1
    metadata: Dict[str, Any] = None

@dataclass
class ServiceRequest:
    """Service request information."""
    request_id: str
    service_type: ServiceType
    method: str
    endpoint: str
    headers: Dict[str, str]
    body: Any
    timeout: int = 30
    retry_count: int = 3
    timestamp: float

@dataclass
class ServiceResponse:
    """Service response information."""
    request_id: str
    status_code: int
    headers: Dict[str, str]
    body: Any
    response_time: float
    service_instance_id: str
    timestamp: float

class ServiceRegistry:
    """Service discovery and registry."""
    
    def __init__(self):
        self.services: Dict[ServiceType, List[ServiceInstance]] = {}
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5  # seconds
        self.health_check_thread: Optional[threading.Thread] = None
        self.registry_active = False
        
        logger.info("Service Registry initialized")
    
    def register_service(self, service_instance: ServiceInstance):
        """Register a service instance."""
        try:
            service_type = service_instance.service_type
            
            if service_type not in self.services:
                self.services[service_type] = []
            
            # Check if instance already exists
            existing_instance = None
            for instance in self.services[service_type]:
                if instance.instance_id == service_instance.instance_id:
                    existing_instance = instance
                    break
            
            if existing_instance:
                # Update existing instance
                existing_instance.status = service_instance.status
                existing_instance.last_health_check = service_instance.last_health_check
                existing_instance.response_time = service_instance.response_time
                existing_instance.cpu_usage = service_instance.cpu_usage
                existing_instance.memory_usage = service_instance.memory_usage
                existing_instance.active_connections = service_instance.active_connections
                existing_instance.metadata = service_instance.metadata
            else:
                # Add new instance
                self.services[service_type].append(service_instance)
            
            logger.info(f"Registered service: {service_type.value} - {service_instance.instance_id}")
            
        except Exception as e:
            logger.error(f"Error registering service: {e}")
    
    def unregister_service(self, service_type: ServiceType, instance_id: str):
        """Unregister a service instance."""
        try:
            if service_type in self.services:
                self.services[service_type] = [
                    instance for instance in self.services[service_type]
                    if instance.instance_id != instance_id
                ]
                
                logger.info(f"Unregistered service: {service_type.value} - {instance_id}")
            
        except Exception as e:
            logger.error(f"Error unregistering service: {e}")
    
    def get_healthy_instances(self, service_type: ServiceType) -> List[ServiceInstance]:
        """Get healthy service instances."""
        try:
            if service_type not in self.services:
                return []
            
            current_time = time.time()
            healthy_instances = []
            
            for instance in self.services[service_type]:
                # Check if instance is healthy and not stale
                if (instance.status == ServiceStatus.HEALTHY and
                    current_time - instance.last_health_check < self.health_check_interval * 2):
                    healthy_instances.append(instance)
            
            return healthy_instances
            
        except Exception as e:
            logger.error(f"Error getting healthy instances: {e}")
            return []
    
    def start_health_checks(self):
        """Start health check monitoring."""
        try:
            if self.registry_active:
                return
            
            self.registry_active = True
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self.health_check_thread.start()
            
            logger.info("Health check monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting health checks: {e}")
    
    def stop_health_checks(self):
        """Stop health check monitoring."""
        self.registry_active = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        
        logger.info("Health check monitoring stopped")
    
    def _health_check_loop(self):
        """Health check monitoring loop."""
        while self.registry_active:
            try:
                for service_type, instances in self.services.items():
                    for instance in instances:
                        self._check_instance_health(instance)
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(self.health_check_interval)
    
    def _check_instance_health(self, instance: ServiceInstance):
        """Check health of a service instance."""
        try:
            start_time = time.time()
            
            # Make health check request
            response = requests.get(
                instance.health_check_url,
                timeout=self.health_check_timeout
            )
            
            response_time = time.time() - start_time
            
            # Update instance status
            if response.status_code == 200:
                instance.status = ServiceStatus.HEALTHY
                instance.response_time = response_time
            else:
                instance.status = ServiceStatus.UNHEALTHY
            
            instance.last_health_check = time.time()
            
        except Exception as e:
            logger.warning(f"Health check failed for {instance.instance_id}: {e}")
            instance.status = ServiceStatus.UNHEALTHY
            instance.last_health_check = time.time()

class LoadBalancer:
    """Advanced load balancer for microservices."""
    
    def __init__(self, service_registry: ServiceRegistry, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.service_registry = service_registry
        self.strategy = strategy
        self.round_robin_counters: Dict[ServiceType, int] = {}
        
        logger.info(f"Load Balancer initialized with {strategy.value} strategy")
    
    def select_instance(self, service_type: ServiceType) -> Optional[ServiceInstance]:
        """Select a service instance using load balancing strategy."""
        try:
            healthy_instances = self.service_registry.get_healthy_instances(service_type)
            
            if not healthy_instances:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(service_type, healthy_instances)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(service_type, healthy_instances)
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return self._random_selection(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.IP_HASH:
                return self._ip_hash_selection(healthy_instances)
            else:
                return healthy_instances[0]
                
        except Exception as e:
            logger.error(f"Error selecting instance: {e}")
            return None
    
    def _round_robin_selection(self, service_type: ServiceType, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection."""
        if service_type not in self.round_robin_counters:
            self.round_robin_counters[service_type] = 0
        
        instance = instances[self.round_robin_counters[service_type] % len(instances)]
        self.round_robin_counters[service_type] += 1
        
        return instance
    
    def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection."""
        return min(instances, key=lambda x: x.active_connections)
    
    def _weighted_round_robin_selection(self, service_type: ServiceType, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin selection."""
        if service_type not in self.round_robin_counters:
            self.round_robin_counters[service_type] = 0
        
        # Calculate total weight
        total_weight = sum(instance.weight for instance in instances)
        
        # Select instance based on weight
        counter = self.round_robin_counters[service_type] % total_weight
        self.round_robin_counters[service_type] += 1
        
        current_weight = 0
        for instance in instances:
            current_weight += instance.weight
            if counter < current_weight:
                return instance
        
        return instances[0]
    
    def _random_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Random selection."""
        return random.choice(instances)
    
    def _ip_hash_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """IP hash selection."""
        # Simplified IP hash - in production, use actual client IP
        client_ip = "127.0.0.1"
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return instances[hash_value % len(instances)]

class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_times: Dict[str, float] = {}
        self.circuit_states: Dict[str, str] = {}  # CLOSED, OPEN, HALF_OPEN
        
        logger.info("Circuit Breaker initialized")
    
    def can_execute(self, service_instance_id: str) -> bool:
        """Check if request can be executed."""
        try:
            current_time = time.time()
            
            # Initialize if not exists
            if service_instance_id not in self.circuit_states:
                self.circuit_states[service_instance_id] = "CLOSED"
                self.failure_counts[service_instance_id] = 0
                self.last_failure_times[service_instance_id] = 0
            
            state = self.circuit_states[service_instance_id]
            
            if state == "CLOSED":
                return True
            elif state == "OPEN":
                # Check if recovery timeout has passed
                if current_time - self.last_failure_times[service_instance_id] > self.recovery_timeout:
                    self.circuit_states[service_instance_id] = "HALF_OPEN"
                    return True
                return False
            elif state == "HALF_OPEN":
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            return True
    
    def record_success(self, service_instance_id: str):
        """Record successful request."""
        try:
            self.failure_counts[service_instance_id] = 0
            self.circuit_states[service_instance_id] = "CLOSED"
            
        except Exception as e:
            logger.error(f"Error recording success: {e}")
    
    def record_failure(self, service_instance_id: str):
        """Record failed request."""
        try:
            self.failure_counts[service_instance_id] += 1
            self.last_failure_times[service_instance_id] = time.time()
            
            if self.failure_counts[service_instance_id] >= self.failure_threshold:
                self.circuit_states[service_instance_id] = "OPEN"
                logger.warning(f"Circuit breaker opened for {service_instance_id}")
            
        except Exception as e:
            logger.error(f"Error recording failure: {e}")

class ServiceClient:
    """Client for making requests to microservices."""
    
    def __init__(self, service_registry: ServiceRegistry, load_balancer: LoadBalancer, circuit_breaker: CircuitBreaker):
        self.service_registry = service_registry
        self.load_balancer = load_balancer
        self.circuit_breaker = circuit_breaker
        self.session = aiohttp.ClientSession()
        
        logger.info("Service Client initialized")
    
    async def make_request(self, service_request: ServiceRequest) -> Optional[ServiceResponse]:
        """Make a request to a microservice."""
        try:
            # Select service instance
            instance = self.load_balancer.select_instance(service_request.service_type)
            if not instance:
                logger.error(f"No healthy instances available for {service_request.service_type.value}")
                return None
            
            # Check circuit breaker
            if not self.circuit_breaker.can_execute(instance.instance_id):
                logger.warning(f"Circuit breaker open for {instance.instance_id}")
                return None
            
            # Build URL
            url = f"http://{instance.host}:{instance.port}{service_request.endpoint}"
            
            # Make request
            start_time = time.time()
            
            async with self.session.request(
                method=service_request.method,
                url=url,
                headers=service_request.headers,
                json=service_request.body,
                timeout=aiohttp.ClientTimeout(total=service_request.timeout)
            ) as response:
                
                response_time = time.time() - start_time
                
                # Read response body
                if response.content_type == 'application/json':
                    body = await response.json()
                else:
                    body = await response.text()
                
                # Create response object
                service_response = ServiceResponse(
                    request_id=service_request.request_id,
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=body,
                    response_time=response_time,
                    service_instance_id=instance.instance_id,
                    timestamp=time.time()
                )
                
                # Record success/failure
                if response.status < 400:
                    self.circuit_breaker.record_success(instance.instance_id)
                else:
                    self.circuit_breaker.record_failure(instance.instance_id)
                
                return service_response
                
        except Exception as e:
            logger.error(f"Error making request: {e}")
            if instance:
                self.circuit_breaker.record_failure(instance.instance_id)
            return None
    
    async def close(self):
        """Close the client session."""
        await self.session.close()

class MicroservicesOrchestrator:
    """Main microservices orchestrator."""
    
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancer(self.service_registry)
        self.circuit_breaker = CircuitBreaker()
        self.service_client = ServiceClient(
            self.service_registry, 
            self.load_balancer, 
            self.circuit_breaker
        )
        self.request_history: List[ServiceRequest] = []
        self.response_history: List[ServiceResponse] = []
        
        logger.info("Microservices Orchestrator initialized")
    
    def start(self):
        """Start the microservices orchestrator."""
        try:
            # Start service registry health checks
            self.service_registry.start_health_checks()
            
            logger.info("Microservices Orchestrator started")
            
        except Exception as e:
            logger.error(f"Error starting orchestrator: {e}")
    
    def stop(self):
        """Stop the microservices orchestrator."""
        try:
            # Stop service registry health checks
            self.service_registry.stop_health_checks()
            
            # Close service client
            asyncio.create_task(self.service_client.close())
            
            logger.info("Microservices Orchestrator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping orchestrator: {e}")
    
    def register_service(self, service_instance: ServiceInstance):
        """Register a service instance."""
        self.service_registry.register_service(service_instance)
    
    def unregister_service(self, service_type: ServiceType, instance_id: str):
        """Unregister a service instance."""
        self.service_registry.unregister_service(service_type, instance_id)
    
    async def call_service(self, service_type: ServiceType, method: str, endpoint: str,
                          headers: Dict[str, str] = None, body: Any = None,
                          timeout: int = 30) -> Optional[ServiceResponse]:
        """Call a microservice."""
        try:
            # Create service request
            request = ServiceRequest(
                request_id=str(uuid.uuid4()),
                service_type=service_type,
                method=method,
                endpoint=endpoint,
                headers=headers or {},
                body=body,
                timeout=timeout
            )
            
            # Store request
            self.request_history.append(request)
            
            # Make request
            response = await self.service_client.make_request(request)
            
            if response:
                self.response_history.append(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling service: {e}")
            return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        try:
            status = {}
            
            for service_type, instances in self.service_registry.services.items():
                healthy_count = len([i for i in instances if i.status == ServiceStatus.HEALTHY])
                total_count = len(instances)
                
                status[service_type.value] = {
                    "total_instances": total_count,
                    "healthy_instances": healthy_count,
                    "unhealthy_instances": total_count - healthy_count,
                    "instances": [asdict(instance) for instance in instances]
                }
            
            return {
                "services": status,
                "total_requests": len(self.request_history),
                "total_responses": len(self.response_history),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {"error": str(e)}

# Global microservices orchestrator instance
_global_microservices: Optional[MicroservicesOrchestrator] = None

def get_microservices_orchestrator() -> MicroservicesOrchestrator:
    """Get the global microservices orchestrator instance."""
    global _global_microservices
    if _global_microservices is None:
        _global_microservices = MicroservicesOrchestrator()
    return _global_microservices

def start_microservices():
    """Start microservices orchestrator."""
    orchestrator = get_microservices_orchestrator()
    orchestrator.start()

def stop_microservices():
    """Stop microservices orchestrator."""
    orchestrator = get_microservices_orchestrator()
    orchestrator.stop()

async def call_microservice(service_type: ServiceType, method: str, endpoint: str,
                           headers: Dict[str, str] = None, body: Any = None,
                           timeout: int = 30) -> Optional[ServiceResponse]:
    """Call a microservice."""
    orchestrator = get_microservices_orchestrator()
    return await orchestrator.call_service(service_type, method, endpoint, headers, body, timeout)

def get_microservices_status() -> Dict[str, Any]:
    """Get microservices status."""
    orchestrator = get_microservices_orchestrator()
    return orchestrator.get_service_status()


