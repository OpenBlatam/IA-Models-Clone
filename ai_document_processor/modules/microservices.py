"""
Microservices Architecture - Ultra-Modular Service System
========================================================

Ultra-modular microservices system with independent service containers and communication.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class ServiceType(str, Enum):
    """Service type enumeration."""
    DOCUMENT_PROCESSOR = "document_processor"
    AI_SERVICE = "ai_service"
    TRANSFORM_SERVICE = "transform_service"
    VALIDATION_SERVICE = "validation_service"
    CACHE_SERVICE = "cache_service"
    FILE_SERVICE = "file_service"
    NOTIFICATION_SERVICE = "notification_service"
    METRICS_SERVICE = "metrics_service"
    API_GATEWAY = "api_gateway"
    MESSAGE_BUS = "message_bus"
    CUSTOM = "custom"


@dataclass
class ServiceConfiguration:
    """Service configuration."""
    name: str
    service_type: ServiceType
    host: str = "localhost"
    port: int = 8000
    health_check_interval: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    dependencies: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    scaling_config: Dict[str, Any] = field(default_factory=dict)
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceInfo:
    """Service information."""
    id: str
    name: str
    service_type: ServiceType
    configuration: ServiceConfiguration
    status: ServiceStatus = ServiceStatus.STOPPED
    health_score: float = 0.0
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    request_count: int = 0
    response_time_avg: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Microservice(ABC):
    """Base microservice class."""
    
    def __init__(self, configuration: ServiceConfiguration):
        self.configuration = configuration
        self.status = ServiceStatus.STOPPED
        self.health_score = 0.0
        self.error_count = 0
        self.last_error = None
        self.start_time = None
        self.stop_time = None
        self.request_count = 0
        self.response_times = []
        self.metadata = {}
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the microservice."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the microservice."""
        pass
    
    @abstractmethod
    async def health_check(self) -> float:
        """Perform health check."""
        pass
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request."""
        pass
    
    def get_service_info(self) -> ServiceInfo:
        """Get service information."""
        return ServiceInfo(
            id=str(uuid.uuid4()),
            name=self.configuration.name,
            service_type=self.configuration.service_type,
            configuration=self.configuration,
            status=self.status,
            health_score=self.health_score,
            last_health_check=datetime.utcnow(),
            error_count=self.error_count,
            last_error=self.last_error,
            start_time=self.start_time,
            stop_time=self.stop_time,
            request_count=self.request_count,
            response_time_avg=sum(self.response_times) / len(self.response_times) if self.response_times else 0.0,
            metadata=self.metadata
        )


class DocumentProcessorMicroservice(Microservice):
    """Document processor microservice."""
    
    def __init__(self, configuration: ServiceConfiguration):
        super().__init__(configuration)
        self.processor = None
    
    async def start(self) -> bool:
        """Start the document processor service."""
        try:
            self.status = ServiceStatus.STARTING
            
            # Initialize document processor
            # self.processor = DocumentProcessor()
            # await self.processor.initialize()
            
            self.status = ServiceStatus.RUNNING
            self.start_time = datetime.utcnow()
            self.health_score = 1.0
            
            logger.info(f"Started document processor service: {self.configuration.name}")
            return True
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Failed to start document processor service: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the document processor service."""
        try:
            self.status = ServiceStatus.STOPPING
            
            # Cleanup processor
            if self.processor:
                await self.processor.cleanup()
            
            self.status = ServiceStatus.STOPPED
            self.stop_time = datetime.utcnow()
            
            logger.info(f"Stopped document processor service: {self.configuration.name}")
            return True
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Failed to stop document processor service: {e}")
            return False
    
    async def health_check(self) -> float:
        """Perform health check."""
        try:
            if self.status != ServiceStatus.RUNNING:
                return 0.0
            
            # Check processor health
            if self.processor and hasattr(self.processor, 'health_check'):
                health_score = await self.processor.health_check()
            else:
                health_score = 1.0
            
            self.health_score = health_score
            return health_score
            
        except Exception as e:
            self.health_score = 0.0
            self.last_error = str(e)
            self.error_count += 1
            return 0.0
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document processing request."""
        try:
            start_time = datetime.utcnow()
            
            # Process the request
            if self.processor:
                result = await self.processor.process_document(request)
            else:
                result = {"status": "error", "message": "Processor not initialized"}
            
            # Record metrics
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            self.response_times.append(response_time)
            self.request_count += 1
            
            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            return {"status": "error", "message": str(e)}


class AIServiceMicroservice(Microservice):
    """AI service microservice."""
    
    def __init__(self, configuration: ServiceConfiguration):
        super().__init__(configuration)
        self.ai_service = None
    
    async def start(self) -> bool:
        """Start the AI service."""
        try:
            self.status = ServiceStatus.STARTING
            
            # Initialize AI service
            # self.ai_service = AIService()
            # await self.ai_service.initialize()
            
            self.status = ServiceStatus.RUNNING
            self.start_time = datetime.utcnow()
            self.health_score = 1.0
            
            logger.info(f"Started AI service: {self.configuration.name}")
            return True
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Failed to start AI service: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the AI service."""
        try:
            self.status = ServiceStatus.STOPPING
            
            # Cleanup AI service
            if self.ai_service:
                await self.ai_service.cleanup()
            
            self.status = ServiceStatus.STOPPED
            self.stop_time = datetime.utcnow()
            
            logger.info(f"Stopped AI service: {self.configuration.name}")
            return True
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Failed to stop AI service: {e}")
            return False
    
    async def health_check(self) -> float:
        """Perform health check."""
        try:
            if self.status != ServiceStatus.RUNNING:
                return 0.0
            
            # Check AI service health
            if self.ai_service and hasattr(self.ai_service, 'health_check'):
                health_score = await self.ai_service.health_check()
            else:
                health_score = 1.0
            
            self.health_score = health_score
            return health_score
            
        except Exception as e:
            self.health_score = 0.0
            self.last_error = str(e)
            self.error_count += 1
            return 0.0
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an AI service request."""
        try:
            start_time = datetime.utcnow()
            
            # Process the request
            if self.ai_service:
                result = await self.ai_service.process_request(request)
            else:
                result = {"status": "error", "message": "AI service not initialized"}
            
            # Record metrics
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            self.response_times.append(response_time)
            self.request_count += 1
            
            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            return {"status": "error", "message": str(e)}


class ServiceContainer:
    """Service container for managing microservices."""
    
    def __init__(self):
        self._services: Dict[str, Microservice] = {}
        self._service_info: Dict[str, ServiceInfo] = {}
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    async def register_service(self, service: Microservice) -> str:
        """Register a microservice."""
        async with self._lock:
            service_id = str(uuid.uuid4())
            self._services[service_id] = service
            self._service_info[service_id] = service.get_service_info()
            
            logger.info(f"Registered service: {service.configuration.name} ({service_id})")
            return service_id
    
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister a microservice."""
        async with self._lock:
            if service_id not in self._services:
                return False
            
            service = self._services[service_id]
            
            # Stop service if running
            if service.status == ServiceStatus.RUNNING:
                await self.stop_service(service_id)
            
            # Remove from storage
            del self._services[service_id]
            del self._service_info[service_id]
            
            # Cancel health check task
            if service_id in self._health_check_tasks:
                self._health_check_tasks[service_id].cancel()
                del self._health_check_tasks[service_id]
            
            logger.info(f"Unregistered service: {service.configuration.name} ({service_id})")
            return True
    
    async def start_service(self, service_id: str) -> bool:
        """Start a microservice."""
        if service_id not in self._services:
            return False
        
        service = self._services[service_id]
        
        try:
            success = await service.start()
            if success:
                # Start health monitoring
                self._health_check_tasks[service_id] = asyncio.create_task(
                    self._health_monitoring_loop(service_id)
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start service {service_id}: {e}")
            return False
    
    async def stop_service(self, service_id: str) -> bool:
        """Stop a microservice."""
        if service_id not in self._services:
            return False
        
        service = self._services[service_id]
        
        try:
            # Stop health monitoring
            if service_id in self._health_check_tasks:
                self._health_check_tasks[service_id].cancel()
                del self._health_check_tasks[service_id]
            
            success = await service.stop()
            return success
            
        except Exception as e:
            logger.error(f"Failed to stop service {service_id}: {e}")
            return False
    
    async def get_service(self, service_id: str) -> Optional[Microservice]:
        """Get service by ID."""
        return self._services.get(service_id)
    
    async def get_service_info(self, service_id: str) -> Optional[ServiceInfo]:
        """Get service info by ID."""
        return self._service_info.get(service_id)
    
    async def get_services_by_type(self, service_type: ServiceType) -> List[Microservice]:
        """Get services by type."""
        return [
            service for service in self._services.values()
            if service.configuration.service_type == service_type
        ]
    
    async def get_running_services(self) -> List[Microservice]:
        """Get all running services."""
        return [
            service for service in self._services.values()
            if service.status == ServiceStatus.RUNNING
        ]
    
    async def process_request(self, service_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through a service."""
        if service_id not in self._services:
            return {"status": "error", "message": "Service not found"}
        
        service = self._services[service_id]
        
        if service.status != ServiceStatus.RUNNING:
            return {"status": "error", "message": "Service not running"}
        
        return await service.process_request(request)
    
    async def _health_monitoring_loop(self, service_id: str):
        """Health monitoring loop for a service."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if service_id not in self._services:
                    break
                
                service = self._services[service_id]
                health_score = await service.health_check()
                
                # Update service info
                if service_id in self._service_info:
                    self._service_info[service_id].health_score = health_score
                    self._service_info[service_id].last_health_check = datetime.utcnow()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error for service {service_id}: {e}")
    
    async def get_container_stats(self) -> Dict[str, Any]:
        """Get container statistics."""
        total_services = len(self._services)
        running_services = len([s for s in self._services.values() if s.status == ServiceStatus.RUNNING])
        error_services = len([s for s in self._services.values() if s.status == ServiceStatus.ERROR])
        
        service_type_counts = {}
        for service_type in ServiceType:
            service_type_counts[service_type.value] = len(
                [s for s in self._services.values() if s.configuration.service_type == service_type]
            )
        
        return {
            'total_services': total_services,
            'running_services': running_services,
            'error_services': error_services,
            'service_type_counts': service_type_counts,
            'health_monitoring_active': len(self._health_check_tasks)
        }


class ServiceDiscovery:
    """Service discovery for microservices."""
    
    def __init__(self):
        self._service_registry: Dict[str, Dict[str, Any]] = {}
        self._service_instances: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
    
    async def register_service_instance(
        self,
        service_name: str,
        instance_id: str,
        host: str,
        port: int,
        metadata: Dict[str, Any] = None
    ):
        """Register a service instance."""
        async with self._lock:
            if service_name not in self._service_instances:
                self._service_instances[service_name] = []
            
            instance_info = {
                'instance_id': instance_id,
                'host': host,
                'port': port,
                'metadata': metadata or {},
                'registered_at': datetime.utcnow(),
                'health_score': 1.0,
                'last_health_check': datetime.utcnow()
            }
            
            self._service_instances[service_name].append(instance_info)
            
            logger.info(f"Registered service instance: {service_name} ({instance_id})")
    
    async def unregister_service_instance(self, service_name: str, instance_id: str):
        """Unregister a service instance."""
        async with self._lock:
            if service_name in self._service_instances:
                self._service_instances[service_name] = [
                    instance for instance in self._service_instances[service_name]
                    if instance['instance_id'] != instance_id
                ]
                
                if not self._service_instances[service_name]:
                    del self._service_instances[service_name]
                
                logger.info(f"Unregistered service instance: {service_name} ({instance_id})")
    
    async def discover_services(self, service_name: str) -> List[Dict[str, Any]]:
        """Discover service instances."""
        return self._service_instances.get(service_name, [])
    
    async def get_healthy_instances(self, service_name: str) -> List[Dict[str, Any]]:
        """Get healthy service instances."""
        instances = await self.discover_services(service_name)
        return [instance for instance in instances if instance['health_score'] >= 0.5]
    
    async def get_service_url(self, service_name: str) -> Optional[str]:
        """Get a service URL (load balanced)."""
        healthy_instances = await self.get_healthy_instances(service_name)
        if not healthy_instances:
            return None
        
        # Simple round-robin load balancing
        instance = healthy_instances[0]  # In production, implement proper load balancing
        return f"http://{instance['host']}:{instance['port']}"


# Global service container instance
_service_container: Optional[ServiceContainer] = None
_service_discovery: Optional[ServiceDiscovery] = None


def get_service_container() -> ServiceContainer:
    """Get global service container instance."""
    global _service_container
    if _service_container is None:
        _service_container = ServiceContainer()
    return _service_container


def get_service_discovery() -> ServiceDiscovery:
    """Get global service discovery instance."""
    global _service_discovery
    if _service_discovery is None:
        _service_discovery = ServiceDiscovery()
    return _service_discovery

















