"""
Microservices Integration for Test Generation System
==================================================

This module provides comprehensive microservices integration capabilities
for distributed test generation across multiple services.
"""

import asyncio
import json
import aiohttp
import redis
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import uuid
import hashlib

from .base_architecture import TestCase, TestGenerationConfig
from .unified_api import TestGenerationAPI, create_api

logger = logging.getLogger(__name__)


@dataclass
class MicroserviceConfig:
    """Configuration for microservices integration"""
    # Service Discovery
    service_registry_url: str = "http://localhost:8500"
    service_name: str = "test-generation-service"
    service_version: str = "1.0.0"
    service_port: int = 8000
    
    # Communication
    api_gateway_url: str = "http://localhost:3000"
    message_broker_url: str = "redis://localhost:6379"
    event_bus_url: str = "redis://localhost:6379"
    
    # Load Balancing
    load_balancer_strategy: str = "round_robin"  # round_robin, least_connections, weighted
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    
    # Caching
    cache_ttl_seconds: int = 3600
    cache_prefix: str = "test_gen"
    distributed_cache: bool = True
    
    # Monitoring
    health_check_interval: int = 30
    metrics_interval: int = 60
    tracing_enabled: bool = True


@dataclass
class ServiceInstance:
    """Service instance information"""
    id: str
    name: str
    version: str
    host: str
    port: int
    health: str = "healthy"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    load: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceRequest:
    """Service request information"""
    id: str
    service_name: str
    method: str
    endpoint: str
    payload: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ServiceResponse:
    """Service response information"""
    request_id: str
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    processing_time: float = 0.0
    service_instance: Optional[ServiceInstance] = None
    error: Optional[str] = None


class ServiceRegistry:
    """Service registry for service discovery"""
    
    def __init__(self, config: MicroserviceConfig):
        self.config = config
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def register_service(self, instance: ServiceInstance) -> bool:
        """Register service instance"""
        try:
            if instance.name not in self.services:
                self.services[instance.name] = []
            
            # Check if instance already exists
            existing = next(
                (s for s in self.services[instance.name] if s.id == instance.id),
                None
            )
            
            if existing:
                # Update existing instance
                existing.host = instance.host
                existing.port = instance.port
                existing.health = instance.health
                existing.last_heartbeat = instance.last_heartbeat
                existing.load = instance.load
                existing.metadata = instance.metadata
            else:
                # Add new instance
                self.services[instance.name].append(instance)
            
            self.logger.info(f"Registered service instance: {instance.name}@{instance.host}:{instance.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register service: {e}")
            return False
    
    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister service instance"""
        try:
            if service_name in self.services:
                self.services[service_name] = [
                    s for s in self.services[service_name] if s.id != instance_id
                ]
                self.logger.info(f"Deregistered service instance: {service_name}#{instance_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to deregister service: {e}")
            return False
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover service instances"""
        try:
            instances = self.services.get(service_name, [])
            
            # Filter healthy instances
            healthy_instances = [
                instance for instance in instances
                if instance.health == "healthy" and
                (datetime.now() - instance.last_heartbeat).seconds < 60
            ]
            
            return healthy_instances
            
        except Exception as e:
            self.logger.error(f"Failed to discover services: {e}")
            return []
    
    async def get_service_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Get best service instance using load balancing strategy"""
        instances = await self.discover_services(service_name)
        
        if not instances:
            return None
        
        if self.config.load_balancer_strategy == "round_robin":
            # Simple round robin (in real implementation, this would be more sophisticated)
            return instances[0]
        elif self.config.load_balancer_strategy == "least_connections":
            # Return instance with least load
            return min(instances, key=lambda x: x.load)
        elif self.config.load_balancer_strategy == "weighted":
            # Return instance with lowest load (simplified)
            return min(instances, key=lambda x: x.load)
        else:
            return instances[0]


class MessageBroker:
    """Message broker for asynchronous communication"""
    
    def __init__(self, config: MicroserviceConfig):
        self.config = config
        self.redis_client = redis.Redis.from_url(config.message_broker_url)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def publish_message(self, topic: str, message: Dict[str, Any]) -> bool:
        """Publish message to topic"""
        try:
            message_data = {
                "id": str(uuid.uuid4()),
                "topic": topic,
                "payload": message,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.redis_client.lpush(f"topic:{topic}", json.dumps(message_data))
            self.logger.debug(f"Published message to topic {topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            return False
    
    async def subscribe_to_topic(self, topic: str, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to topic with callback"""
        try:
            while True:
                # Blocking pop from topic
                message_data = await self.redis_client.brpop(f"topic:{topic}", timeout=1)
                
                if message_data:
                    message = json.loads(message_data[1])
                    await callback(message)
                
        except Exception as e:
            self.logger.error(f"Failed to subscribe to topic: {e}")
    
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Publish event to event bus"""
        try:
            event = {
                "id": str(uuid.uuid4()),
                "type": event_type,
                "data": event_data,
                "timestamp": datetime.now().isoformat(),
                "source": self.config.service_name
            }
            
            await self.redis_client.lpush("events", json.dumps(event))
            self.logger.debug(f"Published event {event_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
            return False


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, config: MicroserviceConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if timeout has passed
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).seconds > self.config.circuit_breaker_timeout:
                self.state = "half_open"
                return True
            return False
        elif self.state == "half_open":
            return True
        
        return False
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
        self.logger.debug("Circuit breaker: Success recorded, state reset to closed")
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.circuit_breaker_threshold:
            self.state = "open"
            self.logger.warning(f"Circuit breaker: Opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class DistributedCache:
    """Distributed cache for microservices"""
    
    def __init__(self, config: MicroserviceConfig):
        self.config = config
        self.redis_client = redis.Redis.from_url(config.message_broker_url)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            cache_key = f"{self.config.cache_prefix}:{key}"
            value = await self.redis_client.get(cache_key)
            
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get from cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            cache_key = f"{self.config.cache_prefix}:{key}"
            ttl = ttl or self.config.cache_ttl_seconds
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(value, default=str)
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            cache_key = f"{self.config.cache_prefix}:{key}"
            await self.redis_client.delete(cache_key)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete from cache: {e}")
            return False


class MicroserviceClient:
    """Microservice client for service communication"""
    
    def __init__(self, config: MicroserviceConfig):
        self.config = config
        self.registry = ServiceRegistry(config)
        self.message_broker = MessageBroker(config)
        self.circuit_breaker = CircuitBreaker(config)
        self.cache = DistributedCache(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def call_service(
        self,
        service_name: str,
        method: str,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> ServiceResponse:
        """Call microservice"""
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            return ServiceResponse(
                request_id=str(uuid.uuid4()),
                status_code=503,
                data={"error": "Circuit breaker is open"},
                error="Service unavailable"
            )
        
        # Get service instance
        instance = await self.registry.get_service_instance(service_name)
        if not instance:
            self.circuit_breaker.record_failure()
            return ServiceResponse(
                request_id=str(uuid.uuid4()),
                status_code=404,
                data={"error": "Service not found"},
                error="Service unavailable"
            )
        
        # Create request
        request = ServiceRequest(
            id=str(uuid.uuid4()),
            service_name=service_name,
            method=method,
            endpoint=endpoint,
            payload=payload,
            headers=headers or {}
        )
        
        # Make HTTP request
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                url = f"http://{instance.host}:{instance.port}{endpoint}"
                
                async with session.request(
                    method=method,
                    url=url,
                    json=payload,
                    headers=request.headers,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:
                    
                    data = await response.json()
                    processing_time = time.time() - start_time
                    
                    # Record success
                    self.circuit_breaker.record_success()
                    
                    return ServiceResponse(
                        request_id=request.id,
                        status_code=response.status,
                        data=data,
                        headers=dict(response.headers),
                        processing_time=processing_time,
                        service_instance=instance
                    )
        
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.logger.error(f"Service call failed: {e}")
            
            return ServiceResponse(
                request_id=request.id,
                status_code=500,
                data={"error": str(e)},
                error=str(e)
            )
    
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Publish event"""
        return await self.message_broker.publish_event(event_type, event_data)
    
    async def subscribe_to_events(self, event_types: List[str], callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to events"""
        for event_type in event_types:
            asyncio.create_task(
                self.message_broker.subscribe_to_topic(f"events:{event_type}", callback)
            )


class DistributedTestGenerator:
    """Distributed test generator for microservices architecture"""
    
    def __init__(self, config: MicroserviceConfig):
        self.config = config
        self.client = MicroserviceClient(config)
        self.api = create_api()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def generate_distributed_tests(
        self,
        function_signature: str,
        docstring: str,
        test_config: Optional[TestGenerationConfig] = None
    ) -> Dict[str, Any]:
        """Generate tests using distributed microservices"""
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(function_signature, docstring, test_config)
            cached_result = await self.client.cache.get(cache_key)
            
            if cached_result:
                self.logger.info("Using cached test generation result")
                return {
                    **cached_result,
                    "from_cache": True
                }
            
            # Publish test generation event
            await self.client.publish_event("test_generation_started", {
                "function_signature": function_signature,
                "docstring": docstring,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate tests using local API
            result = await self.api.generate_tests(
                function_signature,
                docstring,
                "enhanced",
                test_config
            )
            
            if result["success"]:
                # Cache result
                await self.client.cache.set(cache_key, result)
                
                # Publish success event
                await self.client.publish_event("test_generation_completed", {
                    "function_signature": function_signature,
                    "test_count": len(result["test_cases"]),
                    "generation_time": result.get("generation_time", 0),
                    "timestamp": datetime.now().isoformat()
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Distributed test generation failed: {e}")
            
            # Publish error event
            await self.client.publish_event("test_generation_failed", {
                "function_signature": function_signature,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "test_cases": [],
                "error": str(e),
                "success": False
            }
    
    def _generate_cache_key(self, function_signature: str, docstring: str, test_config: Optional[TestGenerationConfig]) -> str:
        """Generate cache key for test generation"""
        key_data = {
            "function_signature": function_signature,
            "docstring": docstring,
            "config": test_config.__dict__ if test_config else {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health information"""
        return {
            "service_name": self.config.service_name,
            "service_version": self.config.service_version,
            "circuit_breaker": self.client.circuit_breaker.get_state(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            "service_name": self.config.service_name,
            "uptime": "unknown",  # Would be calculated in real implementation
            "requests_processed": 0,  # Would be tracked in real implementation
            "cache_hit_rate": 0.0,  # Would be calculated in real implementation
            "circuit_breaker_state": self.client.circuit_breaker.get_state()
        }


# Convenience functions
def create_distributed_generator(config: Optional[MicroserviceConfig] = None) -> DistributedTestGenerator:
    """Create a distributed test generator"""
    if config is None:
        config = MicroserviceConfig()
    return DistributedTestGenerator(config)


async def generate_distributed_tests(
    function_signature: str,
    docstring: str,
    config: Optional[MicroserviceConfig] = None
) -> Dict[str, Any]:
    """Generate tests using distributed microservices"""
    generator = create_distributed_generator(config)
    return await generator.generate_distributed_tests(function_signature, docstring)
