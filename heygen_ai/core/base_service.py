"""
Base Service Classes for HeyGen AI
=================================

Provides base classes and enums for all services in the system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Service type enumeration."""
    CORE = "core"
    API = "api"
    INTEGRATION = "integration"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    STORAGE = "storage"
    NETWORK = "network"
    SECURITY = "security"
    ML = "ml"
    CUSTOM = "custom"


class ServiceStatus(Enum):
    """Service status enumeration."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service_name: str
    status: ServiceStatus
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    
    @property
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self.status in [ServiceStatus.RUNNING, ServiceStatus.STARTING]


class BaseService(ABC):
    """Base class for all services in the system."""
    
    def __init__(self, name: str, service_type: ServiceType = ServiceType.CORE):
        self.name = name
        self.service_type = service_type
        self.status = ServiceStatus.UNKNOWN
        self.start_time: Optional[datetime] = None
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.config: Dict[str, Any] = {}
        self._running = False
        
    @abstractmethod
    async def start(self) -> bool:
        """Start the service."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the service."""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Perform a health check."""
        pass
    
    def get_status(self) -> ServiceStatus:
        """Get current service status."""
        return self.status
    
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running
    
    def get_uptime(self) -> Optional[float]:
        """Get service uptime in seconds."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            "name": self.name,
            "type": self.service_type.value,
            "status": self.status.value,
            "running": self._running,
            "uptime": self.get_uptime(),
            "start_time": self.start_time.isoformat() if self.start_time else None
        }
    
    def update_config(self, config: Dict[str, Any]):
        """Update service configuration."""
        self.config.update(config)
        self.logger.info(f"Configuration updated for {self.name}")
    
    def log_event(self, event: str, level: str = "info", **kwargs):
        """Log an event with structured data."""
        log_data = {
            "service": self.name,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        if level == "debug":
            self.logger.debug(f"Event: {event}", extra=log_data)
        elif level == "info":
            self.logger.info(f"Event: {event}", extra=log_data)
        elif level == "warning":
            self.logger.warning(f"Event: {event}", extra=log_data)
        elif level == "error":
            self.logger.error(f"Event: {event}", extra=log_data)
        elif level == "critical":
            self.logger.critical(f"Event: {event}", extra=log_data)


class ServiceManager:
    """Manages multiple services."""
    
    def __init__(self):
        self.services: Dict[str, BaseService] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_service(self, service: BaseService):
        """Register a service."""
        self.services[service.name] = service
        self.logger.info(f"Service registered: {service.name}")
    
    def unregister_service(self, service_name: str):
        """Unregister a service."""
        if service_name in self.services:
            del self.services[service_name]
            self.logger.info(f"Service unregistered: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[BaseService]:
        """Get a service by name."""
        return self.services.get(service_name)
    
    def get_services_by_type(self, service_type: ServiceType) -> List[BaseService]:
        """Get all services of a specific type."""
        return [s for s in self.services.values() if s.service_type == service_type]
    
    async def start_all_services(self) -> bool:
        """Start all registered services."""
        self.logger.info("Starting all services...")
        success = True
        
        for service in self.services.values():
            try:
                if await service.start():
                    self.logger.info(f"Service started: {service.name}")
                else:
                    self.logger.error(f"Failed to start service: {service.name}")
                    success = False
            except Exception as e:
                self.logger.error(f"Error starting service {service.name}: {e}")
                success = False
        
        return success
    
    async def stop_all_services(self) -> bool:
        """Stop all registered services."""
        self.logger.info("Stopping all services...")
        success = True
        
        for service in self.services.values():
            try:
                if await service.stop():
                    self.logger.info(f"Service stopped: {service.name}")
                else:
                    self.logger.error(f"Failed to stop service: {service.name}")
                    success = False
            except Exception as e:
                self.logger.error(f"Error stopping service {service.name}: {e}")
                success = False
        
        return success
    
    async def health_check_all(self) -> List[HealthCheckResult]:
        """Perform health check on all services."""
        results = []
        
        for service in self.services.values():
            try:
                result = await service.health_check()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Health check failed for {service.name}: {e}")
                results.append(HealthCheckResult(
                    service_name=service.name,
                    status=ServiceStatus.ERROR,
                    error_message=str(e)
                ))
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        total_services = len(self.services)
        running_services = sum(1 for s in self.services.values() if s.is_running())
        healthy_services = sum(1 for s in self.services.values() if s.status == ServiceStatus.RUNNING)
        
        return {
            "total_services": total_services,
            "running_services": running_services,
            "healthy_services": healthy_services,
            "system_health": "healthy" if healthy_services == total_services else "degraded",
            "services": {name: service.get_info() for name, service in self.services.items()}
        }
