"""
Next-Generation Microservices Orchestrator for Facebook Posts
Distributed architecture with service mesh and advanced orchestration
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import aiohttp
import consul
import etcd3
from kubernetes import client, config
import grpc
from concurrent import futures

logger = logging.getLogger(__name__)


# Pure functions for microservices orchestration

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"
    SCALING = "scaling"


class ServiceType(str, Enum):
    API_GATEWAY = "api_gateway"
    CONTENT_SERVICE = "content_service"
    AI_SERVICE = "ai_service"
    ANALYTICS_SERVICE = "analytics_service"
    CACHE_SERVICE = "cache_service"
    SECURITY_SERVICE = "security_service"
    NOTIFICATION_SERVICE = "notification_service"
    WORKFLOW_SERVICE = "workflow_service"


class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASH = "consistent_hash"


@dataclass(frozen=True)
class ServiceInstance:
    """Immutable service instance - pure data structure"""
    service_id: str
    service_type: ServiceType
    host: str
    port: int
    version: str
    status: ServiceStatus
    health_check_url: str
    metadata: Dict[str, Any]
    last_heartbeat: datetime
    load_balancing_weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "service_id": self.service_id,
            "service_type": self.service_type.value,
            "host": self.host,
            "port": self.port,
            "version": self.version,
            "status": self.status.value,
            "health_check_url": self.health_check_url,
            "metadata": self.metadata,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "load_balancing_weight": self.load_balancing_weight
        }


@dataclass(frozen=True)
class ServiceRequest:
    """Immutable service request - pure data structure"""
    request_id: str
    service_type: ServiceType
    method: str
    path: str
    headers: Dict[str, str]
    body: Optional[Any]
    timeout: float
    retry_count: int
    circuit_breaker_enabled: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "request_id": self.request_id,
            "service_type": self.service_type.value,
            "method": self.method,
            "path": self.path,
            "headers": self.headers,
            "body": self.body,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "circuit_breaker_enabled": self.circuit_breaker_enabled
        }


def create_service_instance(
    service_type: ServiceType,
    host: str,
    port: int,
    version: str = "1.0.0",
    metadata: Optional[Dict[str, Any]] = None
) -> ServiceInstance:
    """Create service instance - pure function"""
    return ServiceInstance(
        service_id=f"{service_type.value}_{uuid.uuid4().hex[:8]}",
        service_type=service_type,
        host=host,
        port=port,
        version=version,
        status=ServiceStatus.STARTING,
        health_check_url=f"http://{host}:{port}/health",
        metadata=metadata or {},
        last_heartbeat=datetime.utcnow(),
        load_balancing_weight=1.0
    )


def create_service_request(
    service_type: ServiceType,
    method: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
    timeout: float = 30.0
) -> ServiceRequest:
    """Create service request - pure function"""
    return ServiceRequest(
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        service_type=service_type,
        method=method,
        path=path,
        headers=headers or {},
        body=body,
        timeout=timeout,
        retry_count=0,
        circuit_breaker_enabled=True
    )


def calculate_load_balancing_weight(
    instance: ServiceInstance,
    current_load: float,
    response_time: float,
    error_rate: float
) -> float:
    """Calculate load balancing weight - pure function"""
    # Base weight
    base_weight = instance.load_balancing_weight
    
    # Adjust for current load (lower load = higher weight)
    load_factor = max(0.1, 1.0 - current_load)
    
    # Adjust for response time (lower response time = higher weight)
    response_factor = max(0.1, 1.0 / (1.0 + response_time / 1000.0))
    
    # Adjust for error rate (lower error rate = higher weight)
    error_factor = max(0.1, 1.0 - error_rate)
    
    # Calculate final weight
    final_weight = base_weight * load_factor * response_factor * error_factor
    
    return max(0.1, min(10.0, final_weight))


def select_service_instance(
    instances: List[ServiceInstance],
    strategy: LoadBalancingStrategy,
    request_hash: Optional[str] = None
) -> Optional[ServiceInstance]:
    """Select service instance using load balancing strategy - pure function"""
    if not instances:
        return None
    
    # Filter healthy instances
    healthy_instances = [i for i in instances if i.status == ServiceStatus.HEALTHY]
    if not healthy_instances:
        return None
    
    if strategy == LoadBalancingStrategy.ROUND_ROBIN:
        # Simple round robin (would need state in real implementation)
        return healthy_instances[0]
    
    elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
        # Select instance with least connections
        return min(healthy_instances, key=lambda i: i.metadata.get('active_connections', 0))
    
    elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
        # Select based on weight
        total_weight = sum(i.load_balancing_weight for i in healthy_instances)
        if total_weight == 0:
            return healthy_instances[0]
        
        # Simple weighted selection
        return max(healthy_instances, key=lambda i: i.load_balancing_weight)
    
    elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
        # Select instance with lowest response time
        return min(healthy_instances, key=lambda i: i.metadata.get('avg_response_time', 1000))
    
    elif strategy == LoadBalancingStrategy.CONSISTENT_HASH:
        # Consistent hashing (simplified)
        if request_hash:
            hash_value = hash(request_hash) % len(healthy_instances)
            return healthy_instances[hash_value]
        else:
            return healthy_instances[0]
    
    return healthy_instances[0]


# Microservices Orchestrator Class

class MicroservicesOrchestrator:
    """Next-Generation Microservices Orchestrator with advanced features"""
    
    def __init__(
        self,
        consul_host: str = "localhost",
        consul_port: int = 8500,
        etcd_host: str = "localhost",
        etcd_port: int = 2379,
        kubernetes_config_path: Optional[str] = None
    ):
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.etcd_host = etcd_host
        self.etcd_port = etcd_port
        
        # Service registry
        self.services: Dict[ServiceType, List[ServiceInstance]] = defaultdict(list)
        self.service_instances: Dict[str, ServiceInstance] = {}
        
        # Load balancing
        self.load_balancing_strategies: Dict[ServiceType, LoadBalancingStrategy] = {
            ServiceType.API_GATEWAY: LoadBalancingStrategy.LEAST_RESPONSE_TIME,
            ServiceType.CONTENT_SERVICE: LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
            ServiceType.AI_SERVICE: LoadBalancingStrategy.LEAST_CONNECTIONS,
            ServiceType.ANALYTICS_SERVICE: LoadBalancingStrategy.ROUND_ROBIN,
            ServiceType.CACHE_SERVICE: LoadBalancingStrategy.CONSISTENT_HASH,
            ServiceType.SECURITY_SERVICE: LoadBalancingStrategy.LEAST_RESPONSE_TIME,
            ServiceType.NOTIFICATION_SERVICE: LoadBalancingStrategy.ROUND_ROBIN,
            ServiceType.WORKFLOW_SERVICE: LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
        }
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "failure_count": 0,
            "last_failure_time": None,
            "state": "closed",  # closed, open, half_open
            "failure_threshold": 5,
            "recovery_timeout": 60
        })
        
        # Service discovery clients
        self.consul_client = None
        self.etcd_client = None
        self.k8s_client = None
        
        # HTTP client for service calls
        self.http_client = None
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_trips": 0,
            "service_discoveries": 0,
            "load_balancing_decisions": 0
        }
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.service_discovery_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self) -> None:
        """Start microservices orchestrator"""
        if self.is_running:
            return
        
        try:
            # Initialize service discovery clients
            await self._initialize_service_discovery()
            
            # Initialize HTTP client
            self.http_client = aiohttp.ClientSession()
            
            # Start background tasks
            self.is_running = True
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.service_discovery_task = asyncio.create_task(self._service_discovery_loop())
            
            # Register default services
            await self._register_default_services()
            
            logger.info("Microservices orchestrator started")
            
        except Exception as e:
            logger.error(f"Error starting microservices orchestrator: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop microservices orchestrator"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop background tasks
        if self.health_check_task:
            self.health_check_task.cancel()
        if self.service_discovery_task:
            self.service_discovery_task.cancel()
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.close()
        
        # Close service discovery clients
        if self.etcd_client:
            await self.etcd_client.close()
        
        logger.info("Microservices orchestrator stopped")
    
    async def _initialize_service_discovery(self) -> None:
        """Initialize service discovery clients"""
        try:
            # Initialize Consul client
            self.consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
            
            # Initialize etcd client
            self.etcd_client = etcd3.client(host=self.etcd_host, port=self.etcd_port)
            
            # Initialize Kubernetes client
            try:
                if config.incluster_config():
                    config.load_incluster_config()
                else:
                    config.load_kube_config()
                self.k8s_client = client.CoreV1Api()
            except Exception as e:
                logger.warning(f"Kubernetes client initialization failed: {str(e)}")
            
            logger.info("Service discovery clients initialized")
            
        except Exception as e:
            logger.error(f"Error initializing service discovery: {str(e)}")
            raise
    
    async def _register_default_services(self) -> None:
        """Register default microservices"""
        try:
            # Register API Gateway
            api_gateway = create_service_instance(
                ServiceType.API_GATEWAY,
                "localhost", 8080, "1.0.0",
                {"endpoints": ["/api/v1/*"], "rate_limit": 1000}
            )
            await self.register_service(api_gateway)
            
            # Register Content Service
            content_service = create_service_instance(
                ServiceType.CONTENT_SERVICE,
                "localhost", 8081, "1.0.0",
                {"endpoints": ["/content/*"], "ai_enabled": True}
            )
            await self.register_service(content_service)
            
            # Register AI Service
            ai_service = create_service_instance(
                ServiceType.AI_SERVICE,
                "localhost", 8082, "1.0.0",
                {"models": ["gpt-4", "claude-3", "custom"], "gpu_enabled": True}
            )
            await self.register_service(ai_service)
            
            # Register Analytics Service
            analytics_service = create_service_instance(
                ServiceType.ANALYTICS_SERVICE,
                "localhost", 8083, "1.0.0",
                {"endpoints": ["/analytics/*"], "real_time": True}
            )
            await self.register_service(analytics_service)
            
            # Register Cache Service
            cache_service = create_service_instance(
                ServiceType.CACHE_SERVICE,
                "localhost", 8084, "1.0.0",
                {"type": "redis", "cluster": True}
            )
            await self.register_service(cache_service)
            
            # Register Security Service
            security_service = create_service_instance(
                ServiceType.SECURITY_SERVICE,
                "localhost", 8085, "1.0.0",
                {"threat_detection": True, "compliance": ["GDPR", "CCPA"]}
            )
            await self.register_service(security_service)
            
            logger.info("Default services registered")
            
        except Exception as e:
            logger.error(f"Error registering default services: {str(e)}")
    
    async def register_service(self, instance: ServiceInstance) -> None:
        """Register service instance"""
        try:
            # Add to local registry
            self.services[instance.service_type].append(instance)
            self.service_instances[instance.service_id] = instance
            
            # Register with Consul
            if self.consul_client:
                self.consul_client.agent.service.register(
                    name=instance.service_type.value,
                    service_id=instance.service_id,
                    address=instance.host,
                    port=instance.port,
                    check=consul.Check.http(instance.health_check_url, "10s")
                )
            
            # Register with etcd
            if self.etcd_client:
                key = f"/services/{instance.service_type.value}/{instance.service_id}"
                value = json.dumps(instance.to_dict())
                await self.etcd_client.put(key, value)
            
            self.stats["service_discoveries"] += 1
            logger.info(f"Registered service: {instance.service_type.value} at {instance.host}:{instance.port}")
            
        except Exception as e:
            logger.error(f"Error registering service: {str(e)}")
    
    async def discover_services(self, service_type: ServiceType) -> List[ServiceInstance]:
        """Discover services of specific type"""
        try:
            discovered_instances = []
            
            # Discover from Consul
            if self.consul_client:
                services = self.consul_client.health.service(service_type.value, passing=True)[1]
                for service in services:
                    instance = ServiceInstance(
                        service_id=service['Service']['ID'],
                        service_type=service_type,
                        host=service['Service']['Address'],
                        port=service['Service']['Port'],
                        version=service['Service'].get('Meta', {}).get('version', '1.0.0'),
                        status=ServiceStatus.HEALTHY,
                        health_check_url=f"http://{service['Service']['Address']}:{service['Service']['Port']}/health",
                        metadata=service['Service'].get('Meta', {}),
                        last_heartbeat=datetime.utcnow(),
                        load_balancing_weight=1.0
                    )
                    discovered_instances.append(instance)
            
            # Discover from etcd
            if self.etcd_client:
                key_prefix = f"/services/{service_type.value}/"
                for key, value in self.etcd_client.get_prefix(key_prefix):
                    instance_data = json.loads(value)
                    instance = ServiceInstance(
                        service_id=instance_data['service_id'],
                        service_type=ServiceType(instance_data['service_type']),
                        host=instance_data['host'],
                        port=instance_data['port'],
                        version=instance_data['version'],
                        status=ServiceStatus(instance_data['status']),
                        health_check_url=instance_data['health_check_url'],
                        metadata=instance_data['metadata'],
                        last_heartbeat=datetime.fromisoformat(instance_data['last_heartbeat']),
                        load_balancing_weight=instance_data['load_balancing_weight']
                    )
                    discovered_instances.append(instance)
            
            # Update local registry
            self.services[service_type] = discovered_instances
            for instance in discovered_instances:
                self.service_instances[instance.service_id] = instance
            
            logger.info(f"Discovered {len(discovered_instances)} instances of {service_type.value}")
            return discovered_instances
            
        except Exception as e:
            logger.error(f"Error discovering services: {str(e)}")
            return []
    
    async def call_service(
        self,
        request: ServiceRequest,
        retry_on_failure: bool = True,
        max_retries: int = 3
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Call microservice with load balancing and circuit breaker"""
        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open(request.service_type):
                return False, None, "Circuit breaker is open"
            
            # Get service instances
            instances = self.services.get(request.service_type, [])
            if not instances:
                # Try to discover services
                instances = await self.discover_services(request.service_type)
                if not instances:
                    return False, None, "No service instances available"
            
            # Select instance using load balancing
            strategy = self.load_balancing_strategies.get(request.service_type, LoadBalancingStrategy.ROUND_ROBIN)
            instance = select_service_instance(instances, strategy, request.request_id)
            
            if not instance:
                return False, None, "No healthy service instances available"
            
            # Make HTTP request
            url = f"http://{instance.host}:{instance.port}{request.path}"
            
            async with self.http_client.request(
                method=request.method,
                url=url,
                headers=request.headers,
                json=request.body,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                response_data = await response.json()
                
                if response.status >= 200 and response.status < 300:
                    # Success
                    self._record_success(request.service_type)
                    self.stats["successful_requests"] += 1
                    return True, response_data, None
                else:
                    # HTTP error
                    self._record_failure(request.service_type)
                    self.stats["failed_requests"] += 1
                    return False, None, f"HTTP {response.status}: {response_data.get('error', 'Unknown error')}"
            
        except asyncio.TimeoutError:
            self._record_failure(request.service_type)
            self.stats["failed_requests"] += 1
            return False, None, "Request timeout"
        
        except Exception as e:
            self._record_failure(request.service_type)
            self.stats["failed_requests"] += 1
            return False, None, f"Request failed: {str(e)}"
    
    def _is_circuit_breaker_open(self, service_type: ServiceType) -> bool:
        """Check if circuit breaker is open"""
        circuit_breaker = self.circuit_breakers[service_type.value]
        
        if circuit_breaker["state"] == "open":
            # Check if recovery timeout has passed
            if circuit_breaker["last_failure_time"]:
                time_since_failure = (datetime.utcnow() - circuit_breaker["last_failure_time"]).total_seconds()
                if time_since_failure > circuit_breaker["recovery_timeout"]:
                    circuit_breaker["state"] = "half_open"
                    return False
            return True
        
        return False
    
    def _record_success(self, service_type: ServiceType) -> None:
        """Record successful request"""
        circuit_breaker = self.circuit_breakers[service_type.value]
        circuit_breaker["failure_count"] = 0
        circuit_breaker["state"] = "closed"
    
    def _record_failure(self, service_type: ServiceType) -> None:
        """Record failed request"""
        circuit_breaker = self.circuit_breakers[service_type.value]
        circuit_breaker["failure_count"] += 1
        circuit_breaker["last_failure_time"] = datetime.utcnow()
        
        if circuit_breaker["failure_count"] >= circuit_breaker["failure_threshold"]:
            circuit_breaker["state"] = "open"
            self.stats["circuit_breaker_trips"] += 1
            logger.warning(f"Circuit breaker opened for {service_type.value}")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while self.is_running:
            try:
                for service_type, instances in self.services.items():
                    for instance in instances:
                        await self._check_instance_health(instance)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(10)
    
    async def _check_instance_health(self, instance: ServiceInstance) -> None:
        """Check health of specific instance"""
        try:
            async with self.http_client.get(
                instance.health_check_url,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    # Update instance status
                    updated_instance = ServiceInstance(
                        service_id=instance.service_id,
                        service_type=instance.service_type,
                        host=instance.host,
                        port=instance.port,
                        version=instance.version,
                        status=ServiceStatus.HEALTHY,
                        health_check_url=instance.health_check_url,
                        metadata=instance.metadata,
                        last_heartbeat=datetime.utcnow(),
                        load_balancing_weight=instance.load_balancing_weight
                    )
                    
                    # Update in registry
                    self.service_instances[instance.service_id] = updated_instance
                    for i, existing_instance in enumerate(self.services[instance.service_type]):
                        if existing_instance.service_id == instance.service_id:
                            self.services[instance.service_type][i] = updated_instance
                            break
                
        except Exception as e:
            logger.debug(f"Health check failed for {instance.service_id}: {str(e)}")
            # Mark as unhealthy
            updated_instance = ServiceInstance(
                service_id=instance.service_id,
                service_type=instance.service_type,
                host=instance.host,
                port=instance.port,
                version=instance.version,
                status=ServiceStatus.UNHEALTHY,
                health_check_url=instance.health_check_url,
                metadata=instance.metadata,
                last_heartbeat=instance.last_heartbeat,
                load_balancing_weight=instance.load_balancing_weight
            )
            
            # Update in registry
            self.service_instances[instance.service_id] = updated_instance
            for i, existing_instance in enumerate(self.services[instance.service_type]):
                if existing_instance.service_id == instance.service_id:
                    self.services[instance.service_type][i] = updated_instance
                    break
    
    async def _service_discovery_loop(self) -> None:
        """Background service discovery loop"""
        while self.is_running:
            try:
                for service_type in ServiceType:
                    await self.discover_services(service_type)
                
                await asyncio.sleep(60)  # Discover every minute
                
            except Exception as e:
                logger.error(f"Error in service discovery loop: {str(e)}")
                await asyncio.sleep(30)
    
    def get_service_instances(self, service_type: ServiceType) -> List[ServiceInstance]:
        """Get all instances of a service type"""
        return self.services.get(service_type, [])
    
    def get_healthy_instances(self, service_type: ServiceType) -> List[ServiceInstance]:
        """Get healthy instances of a service type"""
        instances = self.services.get(service_type, [])
        return [i for i in instances if i.status == ServiceStatus.HEALTHY]
    
    def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "statistics": self.stats.copy(),
            "registered_services": {
                service_type.value: len(instances)
                for service_type, instances in self.services.items()
            },
            "circuit_breakers": {
                service_type: {
                    "state": circuit_breaker["state"],
                    "failure_count": circuit_breaker["failure_count"]
                }
                for service_type, circuit_breaker in self.circuit_breakers.items()
            },
            "load_balancing_strategies": {
                service_type.value: strategy.value
                for service_type, strategy in self.load_balancing_strategies.items()
            }
        }


# Factory functions

def create_microservices_orchestrator(
    consul_host: str = "localhost",
    consul_port: int = 8500,
    etcd_host: str = "localhost",
    etcd_port: int = 2379
) -> MicroservicesOrchestrator:
    """Create microservices orchestrator - pure function"""
    return MicroservicesOrchestrator(consul_host, consul_port, etcd_host, etcd_port)


async def get_microservices_orchestrator() -> MicroservicesOrchestrator:
    """Get microservices orchestrator instance"""
    orchestrator = create_microservices_orchestrator()
    await orchestrator.start()
    return orchestrator

