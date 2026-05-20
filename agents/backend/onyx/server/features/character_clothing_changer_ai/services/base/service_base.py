"""
Base Service
============

Base class for all services with lifecycle management.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service state."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BaseService(ABC):
    """Base service class with lifecycle management."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base service.
        
        Args:
            name: Service name
            config: Optional configuration
        """
        self.name = name
        self.config = config or {}
        self.state = ServiceState.UNINITIALIZED
        self.start_time: Optional[float] = None
        self.error: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize service (implemented by subclasses).
        
        Raises:
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    def _cleanup(self) -> None:
        """
        Cleanup service (implemented by subclasses).
        """
        pass
    
    def initialize(self) -> None:
        """
        Initialize the service.
        
        Raises:
            Exception: If initialization fails
        """
        if self.state != ServiceState.UNINITIALIZED:
            logger.warning(f"Service {self.name} already initialized")
            return
        
        self.state = ServiceState.INITIALIZING
        logger.info(f"Initializing service: {self.name}")
        
        try:
            self._initialize()
            self.state = ServiceState.READY
            self.start_time = time.time()
            logger.info(f"Service {self.name} initialized successfully")
        except Exception as e:
            self.state = ServiceState.ERROR
            self.error = str(e)
            logger.error(f"Failed to initialize service {self.name}: {e}")
            raise
    
    def start(self) -> None:
        """Start the service."""
        if self.state == ServiceState.UNINITIALIZED:
            self.initialize()
        
        if self.state != ServiceState.READY:
            raise RuntimeError(f"Service {self.name} is not ready (state: {self.state.value})")
        
        self.state = ServiceState.RUNNING
        logger.info(f"Service {self.name} started")
    
    def stop(self) -> None:
        """Stop the service."""
        if self.state == ServiceState.STOPPED:
            return
        
        self.state = ServiceState.STOPPING
        logger.info(f"Stopping service: {self.name}")
        
        try:
            self._cleanup()
            self.state = ServiceState.STOPPED
            logger.info(f"Service {self.name} stopped")
        except Exception as e:
            self.state = ServiceState.ERROR
            self.error = str(e)
            logger.error(f"Error stopping service {self.name}: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get service status.
        
        Returns:
            Service status dictionary
        """
        uptime = None
        if self.start_time:
            uptime = time.time() - self.start_time
        
        return {
            "name": self.name,
            "state": self.state.value,
            "uptime": uptime,
            "error": self.error,
            "metadata": self.metadata,
        }
    
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self.state == ServiceState.READY
    
    def is_running(self) -> bool:
        """Check if service is running."""
        return self.state == ServiceState.RUNNING
    
    def has_error(self) -> bool:
        """Check if service has error."""
        return self.state == ServiceState.ERROR

