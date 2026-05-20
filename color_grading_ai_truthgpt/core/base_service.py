"""
Base Service for Color Grading AI
==================================

Base class for all services with common functionality.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Base service configuration."""
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 3
    metadata: Dict[str, Any] = None


class BaseService(ABC):
    """
    Base class for all services.
    
    Provides:
    - Common initialization
    - Health checking
    - Statistics tracking
    - Lifecycle management
    """
    
    def __init__(self, name: str, config: Optional[ServiceConfig] = None):
        """
        Initialize base service.
        
        Args:
            name: Service name
            config: Optional service configuration
        """
        self.name = name
        self.config = config or ServiceConfig()
        self._initialized = False
        self._stats: Dict[str, Any] = {
            "calls": 0,
            "errors": 0,
            "last_call": None,
        }
    
    def initialize(self) -> bool:
        """
        Initialize service.
        
        Returns:
            True if successful
        """
        if self._initialized:
            return True
        
        try:
            self._do_initialize()
            self._initialized = True
            logger.info(f"Service {self.name} initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize service {self.name}: {e}")
            return False
    
    @abstractmethod
    def _do_initialize(self):
        """Service-specific initialization."""
        pass
    
    def is_healthy(self) -> bool:
        """
        Check if service is healthy.
        
        Returns:
            True if healthy
        """
        if not self._initialized:
            return False
        
        if not self.config.enabled:
            return False
        
        return self._check_health()
    
    def _check_health(self) -> bool:
        """
        Service-specific health check.
        
        Returns:
            True if healthy
        """
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            **self._stats,
            "name": self.name,
            "initialized": self._initialized,
            "healthy": self.is_healthy(),
            "enabled": self.config.enabled,
        }
    
    def reset_stats(self):
        """Reset service statistics."""
        self._stats = {
            "calls": 0,
            "errors": 0,
            "last_call": None,
        }
    
    def _record_call(self, success: bool = True):
        """Record service call."""
        self._stats["calls"] += 1
        if not success:
            self._stats["errors"] += 1
        from datetime import datetime
        self._stats["last_call"] = datetime.now().isoformat()
    
    async def close(self):
        """Close service and cleanup resources."""
        if self._initialized:
            try:
                await self._do_close()
                self._initialized = False
                logger.info(f"Service {self.name} closed")
            except Exception as e:
                logger.error(f"Error closing service {self.name}: {e}")
    
    async def _do_close(self):
        """Service-specific cleanup."""
        pass




