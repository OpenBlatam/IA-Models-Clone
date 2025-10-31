"""
Microservices Orchestrator for Email Sequence System

This module provides comprehensive microservices orchestration, service discovery,
load balancing, and distributed system management for scalable architecture.
"""

import asyncio
import logging
import time
import json
import aiohttp
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from uuid import UUID
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import consul
import etcd3
from kubernetes import client, config

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .exceptions import OrchestrationError
from .cache import cache_manager

logger = logging.getLogger(__name__)
settings = get_settings()


class ServiceStatus(str, Enum):
    """Service status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    IP_HASH = "ip_hash"


class ServiceDiscoveryBackend(str, Enum):
    """Service discovery backends"""
    CONSUL = "consul"
    ETCD = "etcd"
    KUBERNETES = "kubernetes"
    EUREKA = "eureka"
    ZOOKEEPER = "zookeeper"


@dataclass
class ServiceInstance:
    """Service instance data structure"""
    instance_id: str
    service_name: str
    host: str
    port: int
    status: ServiceStatus
    health_check_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    weight: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None
    response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0


@dataclass
class ServiceDefinition:
    """Service definition"""
    service_name: str
    version: str
    description: str
    endpoints: List[str]
    dependencies: List[str] = field(default_factory=list)
    health_check_interval: int = 30
    timeout: int = 30
    retry_count: int = 3
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state"""
    service_name: str
    state: str  # CLOSED, OPEN, HALF_OPEN
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0
    threshold: int = 5
    timeout: int = 60


class MicroservicesOrchestrator:
    """Microservices orchestrator for distributed system management"""
    
    def __init__(self):
        """Initialize the microservices orchestrator"""
        self.services: Dict[str, ServiceDefinition] = {}
        self.service_instances: Dict[str, List[ServiceInstance]] = defaultdict(list)
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.load_balancers: Dict[str, LoadBalancingStrategy] = {}
        
        # Service discovery
        self.service_discovery_backend: Optional[ServiceDiscoveryBackend] = None
        self.consul_client: Optional[consul.Consul] = None
        self.etcd_client: Optional[etcd3.client] = None
        self.k8s_client: Optional[client.CoreV1Api] = None
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.circuit_breaker_trips = 0
        self.service_discoveries = 0
        
        # Configuration
        self.health_check_interval = 30
        self.service_timeout = 30
        self.retry_count = 3
        self.default_load_balancing_strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        logger.info("Microservices Orchestrator initialized")
    
    async def initialize(self) -> None:
        """Initialize the microservices orchestrator"""
        try:
            # Initialize service discovery backend
            await self._initialize_service_discovery()
            
            # Start background orchestration tasks
            asyncio.create_task(self._service_health_monitor())
            asyncio.create_task(self._service_discovery_monitor())
            asyncio.create_task(self._load_balancer_optimizer())
            asyncio.create_task(self._circuit_breaker_monitor())
            asyncio.create_task(self._service_metrics_collector())
            
            # Register default services
            await self._register_default_services()
            
            logger.info("Microservices Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing microservices orchestrator: {e}")
            raise OrchestrationError(f"Failed to initialize microservices orchestrator: {e}")
    
    async def register_service(self, service_definition: ServiceDefinition) -> bool:
        """
        Register a service with the orchestrator.
        
        Args:
            service_definition: Service definition
            
        Returns:
            True if registered successfully
        """
        try:
            self.services[service_definition.service_name] = service_definition
            self.load_balancers[service_definition.service_name] = service_definition.load_balancing_strategy
            
            # Initialize circuit breaker
            self.circuit_breakers[service_definition.service_name] = CircuitBreakerState(
                service_name=service_definition.service_name,
                state="CLOSED",
                threshold=service_definition.circuit_breaker_threshold,
                timeout=service_definition.circuit_breaker_timeout
            )
            
            # Register with service discovery
            if self.service_discovery_backend:
                await self._register_with_service_discovery(service_definition)
            
            logger.info(f"Service registered: {service_definition.service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering service {service_definition.service_name}: {e}")
            return False
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """
        Discover service instances.
        
        Args:
            service_name: Name of the service to discover
            
        Returns:
            List of service instances
        """
        try:
            self.service_discoveries += 1
            
            # Check if we have cached instances
            if service_name in self.service_instances and self.service_instances[service_name]:
                return self.service_instances[service_name]
            
            # Discover from service discovery backend
            instances = []
            
            if self.service_discovery_backend == ServiceDiscoveryBackend.CONSUL:
                instances = await self._discover_from_consul(service_name)
            elif self.service_discovery_backend == ServiceDiscoveryBackend.ETCD:
                instances = await self._discover_from_etcd(service_name)
            elif self.service_discovery_backend == ServiceDiscoveryBackend.KUBERNETES:
                instances = await self._discover_from_kubernetes(service_name)
            
            # Cache discovered instances
            self.service_instances[service_name] = instances
            
            logger.info(f"Discovered {len(instances)} instances for service: {service_name}")
            return instances
            
        except Exception as e:
            logger.error(f"Error discovering services for {service_name}: {e}")
            return []
    
    async def call_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Call a service with load balancing and circuit breaker.
        
        Args:
            service_name: Name of the service
            endpoint: Service endpoint
            method: HTTP method
            data: Request data
            headers: Request headers
            timeout: Request timeout
            
        Returns:
            Tuple of (success, response_data, error_message)
        """
        try:
            self.total_requests += 1
            
            # Check circuit breaker
            if not await self._check_circuit_breaker(service_name):
                self.failed_requests += 1
                return False, None, "Circuit breaker is open"
            
            # Discover service instances
            instances = await self.discover_services(service_name)
            if not instances:
                self.failed_requests += 1
                return False, None, f"No instances found for service: {service_name}"
            
            # Select instance using load balancing
            selected_instance = await self._select_instance(service_name, instances)
            if not selected_instance:
                self.failed_requests += 1
                return False, None, "No healthy instances available"
            
            # Make request with retry logic
            success, response_data, error_message = await self._make_request_with_retry(
                selected_instance, endpoint, method, data, headers, timeout
            )
            
            # Update circuit breaker and metrics
            if success:
                await self._record_success(service_name, selected_instance)
                self.successful_requests += 1
            else:
                await self._record_failure(service_name, selected_instance)
                self.failed_requests += 1
            
            return success, response_data, error_message
            
        except Exception as e:
            logger.error(f"Error calling service {service_name}: {e}")
            self.failed_requests += 1
            return False, None, str(e)
    
    async def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """
        Get service health information.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service health information
        """
        try:
            instances = await self.discover_services(service_name)
            
            healthy_instances = [i for i in instances if i.status == ServiceStatus.HEALTHY]
            unhealthy_instances = [i for i in instances if i.status == ServiceStatus.UNHEALTHY]
            
            # Calculate health metrics
            total_instances = len(instances)
            healthy_count = len(healthy_instances)
            unhealthy_count = len(unhealthy_instances)
            
            health_percentage = (healthy_count / total_instances * 100) if total_instances > 0 else 0
            
            # Circuit breaker state
            circuit_breaker = self.circuit_breakers.get(service_name)
            circuit_breaker_state = circuit_breaker.state if circuit_breaker else "UNKNOWN"
            
            # Load balancing strategy
            load_balancing_strategy = self.load_balancers.get(service_name, self.default_load_balancing_strategy)
            
            return {
                "service_name": service_name,
                "total_instances": total_instances,
                "healthy_instances": healthy_count,
                "unhealthy_instances": unhealthy_count,
                "health_percentage": health_percentage,
                "circuit_breaker_state": circuit_breaker_state,
                "load_balancing_strategy": load_balancing_strategy.value,
                "instances": [
                    {
                        "instance_id": instance.instance_id,
                        "host": instance.host,
                        "port": instance.port,
                        "status": instance.status.value,
                        "response_time": instance.response_time,
                        "error_count": instance.error_count,
                        "success_count": instance.success_count,
                        "last_health_check": instance.last_health_check.isoformat() if instance.last_health_check else None
                    }
                    for instance in instances
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting service health for {service_name}: {e}")
            return {"error": str(e)}
    
    async def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """
        Get orchestrator metrics.
        
        Returns:
            Orchestrator metrics
        """
        try:
            # Calculate success rate
            success_rate = (
                (self.successful_requests / self.total_requests) * 100
                if self.total_requests > 0 else 0
            )
            
            # Service statistics
            total_services = len(self.services)
            total_instances = sum(len(instances) for instances in self.service_instances.values())
            
            # Circuit breaker statistics
            open_circuit_breakers = len([
                cb for cb in self.circuit_breakers.values() if cb.state == "OPEN"
            ])
            
            return {
                "total_services": total_services,
                "total_instances": total_instances,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "circuit_breaker_trips": self.circuit_breaker_trips,
                "open_circuit_breakers": open_circuit_breakers,
                "service_discoveries": self.service_discoveries,
                "service_discovery_backend": self.service_discovery_backend.value if self.service_discovery_backend else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting orchestrator metrics: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _initialize_service_discovery(self) -> None:
        """Initialize service discovery backend"""
        try:
            # Try to initialize Consul
            try:
                self.consul_client = consul.Consul()
                self.service_discovery_backend = ServiceDiscoveryBackend.CONSUL
                logger.info("Service discovery backend: Consul")
                return
            except Exception:
                pass
            
            # Try to initialize etcd
            try:
                self.etcd_client = etcd3.client()
                self.service_discovery_backend = ServiceDiscoveryBackend.ETCD
                logger.info("Service discovery backend: etcd")
                return
            except Exception:
                pass
            
            # Try to initialize Kubernetes
            try:
                config.load_incluster_config()
                self.k8s_client = client.CoreV1Api()
                self.service_discovery_backend = ServiceDiscoveryBackend.KUBERNETES
                logger.info("Service discovery backend: Kubernetes")
                return
            except Exception:
                pass
            
            logger.warning("No service discovery backend available, using local discovery")
            
        except Exception as e:
            logger.error(f"Error initializing service discovery: {e}")
    
    async def _register_default_services(self) -> None:
        """Register default services"""
        try:
            default_services = [
                ServiceDefinition(
                    service_name="email-sequence-service",
                    version="1.0.0",
                    description="Email sequence management service",
                    endpoints=["/api/v1/email-sequences"],
                    health_check_interval=30,
                    timeout=30,
                    retry_count=3,
                    load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                    circuit_breaker_enabled=True,
                    circuit_breaker_threshold=5,
                    circuit_breaker_timeout=60
                ),
                ServiceDefinition(
                    service_name="ai-service",
                    version="1.0.0",
                    description="AI and machine learning service",
                    endpoints=["/api/v1/ai"],
                    health_check_interval=30,
                    timeout=60,
                    retry_count=3,
                    load_balancing_strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME,
                    circuit_breaker_enabled=True,
                    circuit_breaker_threshold=3,
                    circuit_breaker_timeout=120
                ),
                ServiceDefinition(
                    service_name="analytics-service",
                    version="1.0.0",
                    description="Analytics and reporting service",
                    endpoints=["/api/v1/analytics"],
                    health_check_interval=30,
                    timeout=30,
                    retry_count=3,
                    load_balancing_strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
                    circuit_breaker_enabled=True,
                    circuit_breaker_threshold=5,
                    circuit_breaker_timeout=60
                ),
                ServiceDefinition(
                    service_name="quantum-service",
                    version="1.0.0",
                    description="Quantum computing service",
                    endpoints=["/api/v1/quantum"],
                    health_check_interval=60,
                    timeout=120,
                    retry_count=2,
                    load_balancing_strategy=LoadBalancingStrategy.LEAST_CONNECTIONS,
                    circuit_breaker_enabled=True,
                    circuit_breaker_threshold=3,
                    circuit_breaker_timeout=180
                )
            ]
            
            for service in default_services:
                await self.register_service(service)
            
            logger.info(f"Registered {len(default_services)} default services")
            
        except Exception as e:
            logger.error(f"Error registering default services: {e}")
    
    async def _register_with_service_discovery(self, service_definition: ServiceDefinition) -> None:
        """Register service with service discovery backend"""
        try:
            if self.service_discovery_backend == ServiceDiscoveryBackend.CONSUL:
                await self._register_with_consul(service_definition)
            elif self.service_discovery_backend == ServiceDiscoveryBackend.ETCD:
                await self._register_with_etcd(service_definition)
            elif self.service_discovery_backend == ServiceDiscoveryBackend.KUBERNETES:
                await self._register_with_kubernetes(service_definition)
            
        except Exception as e:
            logger.error(f"Error registering with service discovery: {e}")
    
    async def _register_with_consul(self, service_definition: ServiceDefinition) -> None:
        """Register service with Consul"""
        try:
            # Implement Consul registration
            logger.info(f"Registered {service_definition.service_name} with Consul")
            
        except Exception as e:
            logger.error(f"Error registering with Consul: {e}")
    
    async def _register_with_etcd(self, service_definition: ServiceDefinition) -> None:
        """Register service with etcd"""
        try:
            # Implement etcd registration
            logger.info(f"Registered {service_definition.service_name} with etcd")
            
        except Exception as e:
            logger.error(f"Error registering with etcd: {e}")
    
    async def _register_with_kubernetes(self, service_definition: ServiceDefinition) -> None:
        """Register service with Kubernetes"""
        try:
            # Implement Kubernetes registration
            logger.info(f"Registered {service_definition.service_name} with Kubernetes")
            
        except Exception as e:
            logger.error(f"Error registering with Kubernetes: {e}")
    
    async def _discover_from_consul(self, service_name: str) -> List[ServiceInstance]:
        """Discover services from Consul"""
        try:
            # Implement Consul service discovery
            instances = []
            # This would query Consul for service instances
            return instances
            
        except Exception as e:
            logger.error(f"Error discovering from Consul: {e}")
            return []
    
    async def _discover_from_etcd(self, service_name: str) -> List[ServiceInstance]:
        """Discover services from etcd"""
        try:
            # Implement etcd service discovery
            instances = []
            # This would query etcd for service instances
            return instances
            
        except Exception as e:
            logger.error(f"Error discovering from etcd: {e}")
            return []
    
    async def _discover_from_kubernetes(self, service_name: str) -> List[ServiceInstance]:
        """Discover services from Kubernetes"""
        try:
            # Implement Kubernetes service discovery
            instances = []
            # This would query Kubernetes for service instances
            return instances
            
        except Exception as e:
            logger.error(f"Error discovering from Kubernetes: {e}")
            return []
    
    async def _select_instance(self, service_name: str, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select service instance using load balancing strategy"""
        try:
            # Filter healthy instances
            healthy_instances = [i for i in instances if i.status == ServiceStatus.HEALTHY]
            if not healthy_instances:
                return None
            
            # Get load balancing strategy
            strategy = self.load_balancers.get(service_name, self.default_load_balancing_strategy)
            
            if strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(healthy_instances)
            elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(healthy_instances)
            elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(healthy_instances)
            elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_selection(healthy_instances)
            elif strategy == LoadBalancingStrategy.RANDOM:
                return self._random_selection(healthy_instances)
            elif strategy == LoadBalancingStrategy.IP_HASH:
                return self._ip_hash_selection(healthy_instances)
            else:
                return self._round_robin_selection(healthy_instances)
            
        except Exception as e:
            logger.error(f"Error selecting instance: {e}")
            return None
    
    def _round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin instance selection"""
        if not instances:
            return None
        
        # Simple round robin implementation
        current_time = time.time()
        index = int(current_time) % len(instances)
        return instances[index]
    
    def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections instance selection"""
        if not instances:
            return None
        
        # Select instance with least connections (simplified)
        return min(instances, key=lambda x: x.success_count + x.error_count)
    
    def _weighted_round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin instance selection"""
        if not instances:
            return None
        
        # Weighted selection based on instance weight
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return instances[0]
        
        current_time = time.time()
        weighted_index = int(current_time) % total_weight
        
        current_weight = 0
        for instance in instances:
            current_weight += instance.weight
            if weighted_index < current_weight:
                return instance
        
        return instances[-1]
    
    def _least_response_time_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least response time instance selection"""
        if not instances:
            return None
        
        # Select instance with least response time
        return min(instances, key=lambda x: x.response_time)
    
    def _random_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Random instance selection"""
        if not instances:
            return None
        
        import random
        return random.choice(instances)
    
    def _ip_hash_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """IP hash instance selection"""
        if not instances:
            return None
        
        # Simple hash-based selection
        current_time = time.time()
        hash_value = hash(str(current_time)) % len(instances)
        return instances[hash_value]
    
    async def _make_request_with_retry(
        self,
        instance: ServiceInstance,
        endpoint: str,
        method: str,
        data: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        timeout: Optional[int]
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Make HTTP request with retry logic"""
        try:
            url = f"http://{instance.host}:{instance.port}{endpoint}"
            request_timeout = timeout or self.service_timeout
            
            for attempt in range(self.retry_count):
                try:
                    start_time = time.time()
                    
                    async with aiohttp.ClientSession() as session:
                        if method.upper() == "GET":
                            async with session.get(url, headers=headers, timeout=request_timeout) as response:
                                response_data = await response.json()
                        elif method.upper() == "POST":
                            async with session.post(url, json=data, headers=headers, timeout=request_timeout) as response:
                                response_data = await response.json()
                        elif method.upper() == "PUT":
                            async with session.put(url, json=data, headers=headers, timeout=request_timeout) as response:
                                response_data = await response.json()
                        elif method.upper() == "DELETE":
                            async with session.delete(url, headers=headers, timeout=request_timeout) as response:
                                response_data = await response.json()
                        else:
                            return False, None, f"Unsupported HTTP method: {method}"
                        
                        # Update response time
                        instance.response_time = time.time() - start_time
                        
                        if response.status < 400:
                            return True, response_data, None
                        else:
                            error_message = f"HTTP {response.status}: {response_data.get('error', 'Unknown error')}"
                            if attempt < self.retry_count - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            return False, None, error_message
                
                except asyncio.TimeoutError:
                    error_message = "Request timeout"
                    if attempt < self.retry_count - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False, None, error_message
                
                except Exception as e:
                    error_message = str(e)
                    if attempt < self.retry_count - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False, None, error_message
            
            return False, None, "Max retries exceeded"
            
        except Exception as e:
            logger.error(f"Error making request: {e}")
            return False, None, str(e)
    
    async def _check_circuit_breaker(self, service_name: str) -> bool:
        """Check circuit breaker state"""
        try:
            circuit_breaker = self.circuit_breakers.get(service_name)
            if not circuit_breaker:
                return True
            
            if circuit_breaker.state == "CLOSED":
                return True
            elif circuit_breaker.state == "OPEN":
                # Check if timeout has passed
                if circuit_breaker.last_failure_time:
                    time_since_failure = (datetime.utcnow() - circuit_breaker.last_failure_time).total_seconds()
                    if time_since_failure >= circuit_breaker.timeout:
                        circuit_breaker.state = "HALF_OPEN"
                        return True
                return False
            elif circuit_breaker.state == "HALF_OPEN":
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            return True
    
    async def _record_success(self, service_name: str, instance: ServiceInstance) -> None:
        """Record successful request"""
        try:
            instance.success_count += 1
            
            # Update circuit breaker
            circuit_breaker = self.circuit_breakers.get(service_name)
            if circuit_breaker and circuit_breaker.state == "HALF_OPEN":
                circuit_breaker.success_count += 1
                if circuit_breaker.success_count >= 3:  # Reset threshold
                    circuit_breaker.state = "CLOSED"
                    circuit_breaker.failure_count = 0
                    circuit_breaker.success_count = 0
            
        except Exception as e:
            logger.error(f"Error recording success: {e}")
    
    async def _record_failure(self, service_name: str, instance: ServiceInstance) -> None:
        """Record failed request"""
        try:
            instance.error_count += 1
            
            # Update circuit breaker
            circuit_breaker = self.circuit_breakers.get(service_name)
            if circuit_breaker:
                circuit_breaker.failure_count += 1
                circuit_breaker.last_failure_time = datetime.utcnow()
                
                if circuit_breaker.failure_count >= circuit_breaker.threshold:
                    circuit_breaker.state = "OPEN"
                    self.circuit_breaker_trips += 1
                    logger.warning(f"Circuit breaker opened for service: {service_name}")
            
        except Exception as e:
            logger.error(f"Error recording failure: {e}")
    
    # Background tasks
    async def _service_health_monitor(self) -> None:
        """Background service health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check health of all service instances
                for service_name, instances in self.service_instances.items():
                    for instance in instances:
                        await self._check_instance_health(instance)
                
            except Exception as e:
                logger.error(f"Error in service health monitoring: {e}")
    
    async def _service_discovery_monitor(self) -> None:
        """Background service discovery monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Refresh service discovery for all services
                for service_name in self.services.keys():
                    await self.discover_services(service_name)
                
            except Exception as e:
                logger.error(f"Error in service discovery monitoring: {e}")
    
    async def _load_balancer_optimizer(self) -> None:
        """Background load balancer optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                # Optimize load balancing strategies based on performance
                for service_name, instances in self.service_instances.items():
                    if len(instances) > 1:
                        await self._optimize_load_balancing_strategy(service_name, instances)
                
            except Exception as e:
                logger.error(f"Error in load balancer optimization: {e}")
    
    async def _circuit_breaker_monitor(self) -> None:
        """Background circuit breaker monitoring"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Monitor circuit breaker states
                for service_name, circuit_breaker in self.circuit_breakers.items():
                    if circuit_breaker.state == "OPEN":
                        time_since_failure = (datetime.utcnow() - circuit_breaker.last_failure_time).total_seconds()
                        if time_since_failure >= circuit_breaker.timeout:
                            circuit_breaker.state = "HALF_OPEN"
                            logger.info(f"Circuit breaker moved to HALF_OPEN for service: {service_name}")
                
            except Exception as e:
                logger.error(f"Error in circuit breaker monitoring: {e}")
    
    async def _service_metrics_collector(self) -> None:
        """Background service metrics collection"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                
                # Collect and store service metrics
                await self.get_orchestrator_metrics()
                
            except Exception as e:
                logger.error(f"Error in service metrics collection: {e}")
    
    async def _check_instance_health(self, instance: ServiceInstance) -> None:
        """Check health of a service instance"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{instance.host}:{instance.port}{instance.health_check_url}",
                    timeout=10
                ) as response:
                    if response.status == 200:
                        instance.status = ServiceStatus.HEALTHY
                    else:
                        instance.status = ServiceStatus.UNHEALTHY
                    
                    instance.last_health_check = datetime.utcnow()
                    instance.response_time = time.time() - start_time
            
        except Exception as e:
            instance.status = ServiceStatus.UNHEALTHY
            instance.last_health_check = datetime.utcnow()
            logger.warning(f"Health check failed for {instance.instance_id}: {e}")
    
    async def _optimize_load_balancing_strategy(self, service_name: str, instances: List[ServiceInstance]) -> None:
        """Optimize load balancing strategy based on performance"""
        try:
            # Analyze instance performance
            response_times = [i.response_time for i in instances if i.response_time > 0]
            error_rates = [i.error_count / (i.success_count + i.error_count) for i in instances if (i.success_count + i.error_count) > 0]
            
            if not response_times or not error_rates:
                return
            
            avg_response_time = np.mean(response_times)
            avg_error_rate = np.mean(error_rates)
            
            # Optimize strategy based on metrics
            current_strategy = self.load_balancers.get(service_name, self.default_load_balancing_strategy)
            
            if avg_error_rate > 0.1:  # High error rate
                if current_strategy != LoadBalancingStrategy.LEAST_CONNECTIONS:
                    self.load_balancers[service_name] = LoadBalancingStrategy.LEAST_CONNECTIONS
                    logger.info(f"Optimized load balancing strategy for {service_name}: LEAST_CONNECTIONS")
            elif avg_response_time > 1000:  # High response time
                if current_strategy != LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                    self.load_balancers[service_name] = LoadBalancingStrategy.LEAST_RESPONSE_TIME
                    logger.info(f"Optimized load balancing strategy for {service_name}: LEAST_RESPONSE_TIME")
            
        except Exception as e:
            logger.error(f"Error optimizing load balancing strategy: {e}")


# Global microservices orchestrator instance
microservices_orchestrator = MicroservicesOrchestrator()





























