"""
Advanced Service Base
=====================

Advanced base classes for service implementations.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ServiceState(Enum):
    """Service state."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ServiceMetrics:
    """Service metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    last_request: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)


class AdvancedServiceBase(ABC):
    """Advanced base class for services."""
    
    def __init__(self, name: str):
        """
        Initialize service.
        
        Args:
            name: Service name
        """
        self.name = name
        self.state = ServiceState.IDLE
        self.metrics = ServiceMetrics()
        self.config: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._start_time: Optional[datetime] = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize service."""
        pass
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute service operation."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        pass
    
    async def start(self) -> None:
        """Start service."""
        async with self._lock:
            if self.state != ServiceState.IDLE:
                raise RuntimeError(f"Service {self.name} is not in IDLE state")
            
            try:
                self.state = ServiceState.INITIALIZING
                await self.initialize()
                self.state = ServiceState.READY
                self._start_time = datetime.now()
                logger.info(f"Service {self.name} started")
            except Exception as e:
                self.state = ServiceState.ERROR
                logger.error(f"Error starting service {self.name}: {e}")
                raise
    
    async def stop(self) -> None:
        """Stop service."""
        async with self._lock:
            if self.state == ServiceState.STOPPED:
                return
            
            try:
                self.state = ServiceState.STOPPING
                await self.cleanup()
                self.state = ServiceState.STOPPED
                logger.info(f"Service {self.name} stopped")
            except Exception as e:
                self.state = ServiceState.ERROR
                logger.error(f"Error stopping service {self.name}: {e}")
    
    async def pause(self) -> None:
        """Pause service."""
        async with self._lock:
            if self.state == ServiceState.RUNNING:
                self.state = ServiceState.PAUSED
                logger.info(f"Service {self.name} paused")
    
    async def resume(self) -> None:
        """Resume service."""
        async with self._lock:
            if self.state == ServiceState.PAUSED:
                self.state = ServiceState.RUNNING
                logger.info(f"Service {self.name} resumed")
    
    def update_metrics(self, success: bool, duration: float, error: Optional[str] = None):
        """
        Update service metrics.
        
        Args:
            success: Whether operation was successful
            duration: Operation duration
            error: Optional error message
        """
        self.metrics.total_requests += 1
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            if error:
                self.metrics.errors.append(error)
                if len(self.metrics.errors) > 100:
                    self.metrics.errors = self.metrics.errors[-100:]
        
        self.metrics.total_duration += duration
        self.metrics.avg_duration = self.metrics.total_duration / self.metrics.total_requests
        self.metrics.min_duration = min(self.metrics.min_duration, duration)
        self.metrics.max_duration = max(self.metrics.max_duration, duration)
        self.metrics.last_request = datetime.now()
    
    def get_metrics(self) -> ServiceMetrics:
        """Get service metrics."""
        return self.metrics
    
    def get_uptime(self) -> Optional[float]:
        """Get service uptime in seconds."""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return None
    
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self.state == ServiceState.READY
    
    def is_running(self) -> bool:
        """Check if service is running."""
        return self.state == ServiceState.RUNNING


class ServiceRegistry:
    """Registry for services."""
    
    def __init__(self):
        """Initialize service registry."""
        self.services: Dict[str, AdvancedServiceBase] = {}
    
    def register(self, service: AdvancedServiceBase):
        """
        Register a service.
        
        Args:
            service: Service instance
        """
        self.services[service.name] = service
        logger.info(f"Registered service: {service.name}")
    
    def get(self, name: str) -> Optional[AdvancedServiceBase]:
        """Get service by name."""
        return self.services.get(name)
    
    def get_all(self) -> List[AdvancedServiceBase]:
        """Get all services."""
        return list(self.services.values())
    
    async def start_all(self):
        """Start all services."""
        for service in self.services.values():
            if service.state == ServiceState.IDLE:
                await service.start()
    
    async def stop_all(self):
        """Stop all services."""
        for service in reversed(list(self.services.values())):
            if service.state != ServiceState.STOPPED:
                await service.stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all services."""
        return {
            name: {
                "state": service.state.value,
                "metrics": {
                    "total_requests": service.metrics.total_requests,
                    "successful_requests": service.metrics.successful_requests,
                    "failed_requests": service.metrics.failed_requests,
                    "avg_duration": service.metrics.avg_duration,
                    "uptime": service.get_uptime()
                }
            }
            for name, service in self.services.items()
        }




