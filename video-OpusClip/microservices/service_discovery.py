#!/usr/bin/env python3
"""
Service Discovery and Registry System

Advanced service discovery with:
- Service registration and discovery
- Health checking and monitoring
- Load balancing and failover
- Service mesh integration
- Distributed configuration
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Set
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import aiohttp
from collections import defaultdict, deque

logger = structlog.get_logger("service_discovery")

# =============================================================================
# SERVICE DISCOVERY MODELS
# =============================================================================

class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

class ServiceType(Enum):
    """Service type enumeration."""
    API = "api"
    PROCESSOR = "processor"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    ANALYTICS = "analytics"
    MONITORING = "monitoring"

@dataclass
class ServiceInstance:
    """Service instance information."""
    service_id: str
    service_name: str
    service_type: ServiceType
    host: str
    port: int
    version: str
    status: ServiceStatus
    health_check_url: str
    metadata: Dict[str, Any]
    tags: Set[str]
    registered_at: datetime
    last_health_check: datetime
    weight: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_id": self.service_id,
            "service_name": self.service_name,
            "service_type": self.service_type.value,
            "host": self.host,
            "port": self.port,
            "version": self.version,
            "status": self.status.value,
            "health_check_url": self.health_check_url,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "registered_at": self.registered_at.isoformat(),
            "last_health_check": self.last_health_check.isoformat(),
            "weight": self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ServiceInstance:
        """Create from dictionary."""
        return cls(
            service_id=data["service_id"],
            service_name=data["service_name"],
            service_type=ServiceType(data["service_type"]),
            host=data["host"],
            port=data["port"],
            version=data["version"],
            status=ServiceStatus(data["status"]),
            health_check_url=data["health_check_url"],
            metadata=data["metadata"],
            tags=set(data["tags"]),
            registered_at=datetime.fromisoformat(data["registered_at"]),
            last_health_check=datetime.fromisoformat(data["last_health_check"]),
            weight=data.get("weight", 1)
        )

@dataclass
class ServiceRegistration:
    """Service registration request."""
    service_name: str
    service_type: ServiceType
    host: str
    port: int
    version: str
    health_check_url: str
    metadata: Dict[str, Any]
    tags: Set[str]
    ttl: int = 30  # Time to live in seconds
    weight: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "service_type": self.service_type.value,
            "host": self.host,
            "port": self.port,
            "version": self.version,
            "health_check_url": self.health_check_url,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "ttl": self.ttl,
            "weight": self.weight
        }

@dataclass
class ServiceQuery:
    """Service discovery query."""
    service_name: Optional[str] = None
    service_type: Optional[ServiceType] = None
    tags: Optional[Set[str]] = None
    version: Optional[str] = None
    healthy_only: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "service_type": self.service_type.value if self.service_type else None,
            "tags": list(self.tags) if self.tags else None,
            "version": self.version,
            "healthy_only": self.healthy_only
        }

# =============================================================================
# SERVICE REGISTRY
# =============================================================================

class ServiceRegistry:
    """Advanced service registry with health checking and monitoring."""
    
    def __init__(self, health_check_interval: int = 30, ttl_cleanup_interval: int = 60):
        self.services: Dict[str, ServiceInstance] = {}
        self.service_index: Dict[str, Set[str]] = defaultdict(set)
        self.health_check_interval = health_check_interval
        self.ttl_cleanup_interval = ttl_cleanup_interval
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_registrations': 0,
            'total_deregistrations': 0,
            'health_check_failures': 0,
            'ttl_expirations': 0,
            'last_cleanup': None
        }
    
    async def start(self) -> None:
        """Start the service registry."""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_services())
        
        logger.info("Service registry started")
    
    async def stop(self) -> None:
        """Stop the service registry."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        logger.info("Service registry stopped")
    
    async def register_service(self, registration: ServiceRegistration) -> str:
        """Register a new service."""
        service_id = str(uuid.uuid4())
        
        # Create service instance
        service_instance = ServiceInstance(
            service_id=service_id,
            service_name=registration.service_name,
            service_type=registration.service_type,
            host=registration.host,
            port=registration.port,
            version=registration.version,
            status=ServiceStatus.STARTING,
            health_check_url=registration.health_check_url,
            metadata=registration.metadata,
            tags=registration.tags,
            registered_at=datetime.utcnow(),
            last_health_check=datetime.utcnow(),
            weight=registration.weight
        )
        
        # Register service
        self.services[service_id] = service_instance
        
        # Update indexes
        self.service_index[registration.service_name].add(service_id)
        self.service_index[f"{registration.service_type.value}"].add(service_id)
        
        for tag in registration.tags:
            self.service_index[f"tag:{tag}"].add(service_id)
        
        # Start health checking
        self.health_check_tasks[service_id] = asyncio.create_task(
            self._health_check_service(service_id, registration.ttl)
        )
        
        # Update statistics
        self.stats['total_registrations'] += 1
        
        logger.info(
            "Service registered",
            service_id=service_id,
            service_name=registration.service_name,
            service_type=registration.service_type.value,
            host=registration.host,
            port=registration.port
        )
        
        return service_id
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service."""
        if service_id not in self.services:
            return False
        
        service = self.services[service_id]
        
        # Cancel health check task
        if service_id in self.health_check_tasks:
            self.health_check_tasks[service_id].cancel()
            del self.health_check_tasks[service_id]
        
        # Remove from indexes
        self.service_index[service.service_name].discard(service_id)
        self.service_index[f"{service.service_type.value}"].discard(service_id)
        
        for tag in service.tags:
            self.service_index[f"tag:{tag}"].discard(service_id)
        
        # Remove service
        del self.services[service_id]
        
        # Update statistics
        self.stats['total_deregistrations'] += 1
        
        logger.info(
            "Service deregistered",
            service_id=service_id,
            service_name=service.service_name
        )
        
        return True
    
    async def discover_services(self, query: ServiceQuery) -> List[ServiceInstance]:
        """Discover services based on query."""
        matching_services = set()
        
        # Filter by service name
        if query.service_name:
            matching_services.update(self.service_index[query.service_name])
        
        # Filter by service type
        if query.service_type:
            matching_services.intersection_update(self.service_index[query.service_type.value])
        
        # Filter by tags
        if query.tags:
            for tag in query.tags:
                matching_services.intersection_update(self.service_index[f"tag:{tag}"])
        
        # Get service instances
        services = []
        for service_id in matching_services:
            if service_id in self.services:
                service = self.services[service_id]
                
                # Filter by version
                if query.version and service.version != query.version:
                    continue
                
                # Filter by health status
                if query.healthy_only and service.status != ServiceStatus.HEALTHY:
                    continue
                
                services.append(service)
        
        return services
    
    async def get_service(self, service_id: str) -> Optional[ServiceInstance]:
        """Get a specific service by ID."""
        return self.services.get(service_id)
    
    async def get_services_by_name(self, service_name: str, healthy_only: bool = True) -> List[ServiceInstance]:
        """Get all services with a specific name."""
        query = ServiceQuery(service_name=service_name, healthy_only=healthy_only)
        return await self.discover_services(query)
    
    async def get_services_by_type(self, service_type: ServiceType, healthy_only: bool = True) -> List[ServiceInstance]:
        """Get all services of a specific type."""
        query = ServiceQuery(service_type=service_type, healthy_only=healthy_only)
        return await self.discover_services(query)
    
    async def _health_check_service(self, service_id: str, ttl: int) -> None:
        """Health check a service."""
        while self.is_running and service_id in self.services:
            try:
                service = self.services[service_id]
                
                # Perform health check
                is_healthy = await self._perform_health_check(service)
                
                # Update service status
                if is_healthy:
                    service.status = ServiceStatus.HEALTHY
                else:
                    service.status = ServiceStatus.UNHEALTHY
                    self.stats['health_check_failures'] += 1
                
                service.last_health_check = datetime.utcnow()
                
                # Check TTL
                if (datetime.utcnow() - service.registered_at).total_seconds() > ttl:
                    logger.warning(
                        "Service TTL expired",
                        service_id=service_id,
                        service_name=service.service_name
                    )
                    await self.deregister_service(service_id)
                    self.stats['ttl_expirations'] += 1
                    break
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Health check error",
                    service_id=service_id,
                    error=str(e)
                )
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_check(self, service: ServiceInstance) -> bool:
        """Perform actual health check on a service."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                url = f"http://{service.host}:{service.port}{service.health_check_url}"
                async with session.get(url) as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(
                "Health check failed",
                service_id=service.service_id,
                service_name=service.service_name,
                error=str(e)
            )
            return False
    
    async def _cleanup_expired_services(self) -> None:
        """Clean up expired services."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                expired_services = []
                
                for service_id, service in self.services.items():
                    # Check if service hasn't been health checked recently
                    if (current_time - service.last_health_check).total_seconds() > 300:  # 5 minutes
                        expired_services.append(service_id)
                
                # Remove expired services
                for service_id in expired_services:
                    await self.deregister_service(service_id)
                    self.stats['ttl_expirations'] += 1
                
                self.stats['last_cleanup'] = current_time.isoformat()
                
                await asyncio.sleep(self.ttl_cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup error", error=str(e))
                await asyncio.sleep(self.ttl_cleanup_interval)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            **self.stats,
            'total_services': len(self.services),
            'healthy_services': len([s for s in self.services.values() if s.status == ServiceStatus.HEALTHY]),
            'unhealthy_services': len([s for s in self.services.values() if s.status == ServiceStatus.UNHEALTHY]),
            'active_health_checks': len(self.health_check_tasks)
        }

# =============================================================================
# SERVICE DISCOVERY CLIENT
# =============================================================================

class ServiceDiscoveryClient:
    """Service discovery client for service consumers."""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.service_cache: Dict[str, List[ServiceInstance]] = {}
        self.cache_ttl = 30  # seconds
        self.cache_timestamps: Dict[str, datetime] = {}
    
    async def discover_service(self, service_name: str, version: Optional[str] = None) -> Optional[ServiceInstance]:
        """Discover a single service instance."""
        services = await self.discover_services(service_name, version=version)
        return services[0] if services else None
    
    async def discover_services(self, service_name: str, version: Optional[str] = None, 
                              healthy_only: bool = True) -> List[ServiceInstance]:
        """Discover multiple service instances with caching."""
        cache_key = f"{service_name}:{version}:{healthy_only}"
        
        # Check cache
        if cache_key in self.service_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl:
                return self.service_cache[cache_key]
        
        # Query registry
        query = ServiceQuery(
            service_name=service_name,
            version=version,
            healthy_only=healthy_only
        )
        
        services = await self.registry.discover_services(query)
        
        # Update cache
        self.service_cache[cache_key] = services
        self.cache_timestamps[cache_key] = datetime.utcnow()
        
        return services
    
    async def get_service_url(self, service_name: str, version: Optional[str] = None) -> Optional[str]:
        """Get service URL for HTTP requests."""
        service = await self.discover_service(service_name, version)
        if service:
            return f"http://{service.host}:{service.port}"
        return None
    
    async def get_service_endpoint(self, service_name: str, endpoint: str, 
                                 version: Optional[str] = None) -> Optional[str]:
        """Get full service endpoint URL."""
        base_url = await self.get_service_url(service_name, version)
        if base_url:
            return f"{base_url}{endpoint}"
        return None
    
    def clear_cache(self) -> None:
        """Clear service cache."""
        self.service_cache.clear()
        self.cache_timestamps.clear()

# =============================================================================
# LOAD BALANCER FOR SERVICES
# =============================================================================

class ServiceLoadBalancer:
    """Load balancer for service instances."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.current_index = 0
        self.instance_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'requests': 0,
            'errors': 0,
            'response_times': deque(maxlen=100)
        })
    
    def select_instance(self, services: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select a service instance based on load balancing strategy."""
        if not services:
            return None
        
        # Filter healthy services
        healthy_services = [s for s in services if s.status == ServiceStatus.HEALTHY]
        if not healthy_services:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin(healthy_services)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin(healthy_services)
        elif self.strategy == "least_connections":
            return self._least_connections(healthy_services)
        elif self.strategy == "least_response_time":
            return self._least_response_time(healthy_services)
        else:
            return self._round_robin(healthy_services)
    
    def _round_robin(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection."""
        service = services[self.current_index % len(services)]
        self.current_index += 1
        return service
    
    def _weighted_round_robin(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin selection."""
        total_weight = sum(service.weight for service in services)
        if total_weight == 0:
            return services[0]
        
        # Find service with lowest request count relative to weight
        best_service = services[0]
        best_score = float('inf')
        
        for service in services:
            stats = self.instance_stats[service.service_id]
            score = stats['requests'] / service.weight
            
            if score < best_score:
                best_score = score
                best_service = service
        
        return best_service
    
    def _least_connections(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection."""
        return min(services, key=lambda s: self.instance_stats[s.service_id]['requests'])
    
    def _least_response_time(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Least response time selection."""
        def get_avg_response_time(service):
            response_times = self.instance_stats[service.service_id]['response_times']
            return sum(response_times) / len(response_times) if response_times else 0
        
        return min(services, key=get_avg_response_time)
    
    def record_request(self, service_id: str, response_time: float, success: bool = True) -> None:
        """Record request statistics."""
        stats = self.instance_stats[service_id]
        stats['requests'] += 1
        stats['response_times'].append(response_time)
        
        if not success:
            stats['errors'] += 1
    
    def get_instance_stats(self, service_id: str) -> Dict[str, Any]:
        """Get instance statistics."""
        stats = self.instance_stats[service_id]
        response_times = list(stats['response_times'])
        
        return {
            'requests': stats['requests'],
            'errors': stats['errors'],
            'error_rate': stats['errors'] / stats['requests'] if stats['requests'] > 0 else 0,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0
        }

# =============================================================================
# GLOBAL SERVICE DISCOVERY INSTANCES
# =============================================================================

# Global service registry
service_registry = ServiceRegistry()

# Global service discovery client
service_discovery_client = ServiceDiscoveryClient(service_registry)

# Global service load balancer
service_load_balancer = ServiceLoadBalancer("weighted_round_robin")

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ServiceStatus',
    'ServiceType',
    'ServiceInstance',
    'ServiceRegistration',
    'ServiceQuery',
    'ServiceRegistry',
    'ServiceDiscoveryClient',
    'ServiceLoadBalancer',
    'service_registry',
    'service_discovery_client',
    'service_load_balancer'
]





























