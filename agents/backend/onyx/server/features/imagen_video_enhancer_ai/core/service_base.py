"""
Service Base
============

Base classes for all service types.
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


class ServiceStatus(Enum):
    """Service status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ServiceConfig:
    """Service configuration."""
    name: str
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceResult:
    """Service execution result."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseService(ABC):
    """Base service interface."""
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize service.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self.status = ServiceStatus.IDLE
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_duration": 0.0
        }
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> ServiceResult:
        """
        Execute service operation.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Service result
        """
        pass
    
    async def start(self):
        """Start service."""
        async with self._lock:
            if self.status == ServiceStatus.STOPPED:
                self.status = ServiceStatus.IDLE
                logger.info(f"Service {self.config.name} started")
    
    async def stop(self):
        """Stop service."""
        async with self._lock:
            self.status = ServiceStatus.STOPPED
            logger.info(f"Service {self.config.name} stopped")
    
    async def pause(self):
        """Pause service."""
        async with self._lock:
            if self.status == ServiceStatus.RUNNING:
                self.status = ServiceStatus.PAUSED
                logger.info(f"Service {self.config.name} paused")
    
    async def resume(self):
        """Resume service."""
        async with self._lock:
            if self.status == ServiceStatus.PAUSED:
                self.status = ServiceStatus.IDLE
                logger.info(f"Service {self.config.name} resumed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        total = self.stats["total_requests"]
        success_rate = (
            self.stats["successful_requests"] / total
            if total > 0 else 0.0
        )
        avg_duration = (
            self.stats["total_duration"] / total
            if total > 0 else 0.0
        )
        
        return {
            "name": self.config.name,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "total_requests": total,
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            **self.stats
        }
    
    def _update_stats(self, result: ServiceResult):
        """Update service statistics."""
        self.stats["total_requests"] += 1
        if result.success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        self.stats["total_duration"] += result.duration


class AsyncService(BaseService):
    """Async service implementation."""
    
    def __init__(self, config: ServiceConfig, handler: Callable[..., Awaitable[Any]]):
        """
        Initialize async service.
        
        Args:
            config: Service configuration
            handler: Async handler function
        """
        super().__init__(config)
        self.handler = handler
    
    async def execute(self, *args, **kwargs) -> ServiceResult:
        """Execute service with handler."""
        if not self.config.enabled:
            return ServiceResult(
                success=False,
                error="Service is disabled"
            )
        
        if self.status == ServiceStatus.STOPPED:
            return ServiceResult(
                success=False,
                error="Service is stopped"
            )
        
        start = datetime.now()
        
        try:
            async with self._lock:
                if self.status == ServiceStatus.IDLE:
                    self.status = ServiceStatus.RUNNING
            
            if self.config.timeout:
                result_data = await asyncio.wait_for(
                    self.handler(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result_data = await self.handler(*args, **kwargs)
            
            duration = (datetime.now() - start).total_seconds()
            result = ServiceResult(
                success=True,
                data=result_data,
                duration=duration
            )
            
            async with self._lock:
                if self.status == ServiceStatus.RUNNING:
                    self.status = ServiceStatus.IDLE
            
            self._update_stats(result)
            return result
            
        except asyncio.TimeoutError:
            duration = (datetime.now() - start).total_seconds()
            result = ServiceResult(
                success=False,
                error=f"Service timeout after {self.config.timeout}s",
                duration=duration
            )
            async with self._lock:
                if self.status == ServiceStatus.RUNNING:
                    self.status = ServiceStatus.IDLE
            self._update_stats(result)
            return result
            
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            result = ServiceResult(
                success=False,
                error=str(e),
                duration=duration
            )
            async with self._lock:
                if self.status == ServiceStatus.RUNNING:
                    self.status = ServiceStatus.ERROR
            self._update_stats(result)
            return result


class ServiceRegistry:
    """Registry for services."""
    
    def __init__(self):
        """Initialize service registry."""
        self.services: Dict[str, BaseService] = {}
    
    def register(self, service: BaseService):
        """
        Register a service.
        
        Args:
            service: Service instance
        """
        self.services[service.config.name] = service
        logger.debug(f"Registered service: {service.config.name}")
    
    def get(self, name: str) -> Optional[BaseService]:
        """
        Get service by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance or None
        """
        return self.services.get(name)
    
    async def start_all(self):
        """Start all services."""
        for service in self.services.values():
            await service.start()
    
    async def stop_all(self):
        """Stop all services."""
        for service in self.services.values():
            await service.stop()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all services."""
        return {
            name: service.get_stats()
            for name, service in self.services.items()
        }




