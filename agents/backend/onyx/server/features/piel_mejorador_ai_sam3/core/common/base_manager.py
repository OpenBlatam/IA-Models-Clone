"""
Base Manager for Piel Mejorador AI SAM3
=======================================

Base class for managers with common functionality.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ManagerStats:
    """Base manager statistics."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations


class BaseManager(ABC):
    """
    Base class for managers.
    
    Provides common functionality:
    - Statistics tracking
    - Error handling
    - Resource management
    - Lifecycle management
    """
    
    def __init__(self, name: str):
        """
        Initialize base manager.
        
        Args:
            name: Manager name
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._stats = ManagerStats()
        self._running = False
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the manager."""
        if self._running:
            self.logger.warning(f"{self.name} is already running")
            return
        
        self._running = True
        self.logger.info(f"{self.name} started")
    
    async def stop(self):
        """Stop the manager."""
        if not self._running:
            return
        
        self._running = False
        self.logger.info(f"{self.name} stopped")
    
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running
    
    def record_success(self):
        """Record successful operation."""
        self._stats.total_operations += 1
        self._stats.successful_operations += 1
    
    def record_failure(self):
        """Record failed operation."""
        self._stats.total_operations += 1
        self._stats.failed_operations += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "name": self.name,
            "running": self._running,
            "total_operations": self._stats.total_operations,
            "successful_operations": self._stats.successful_operations,
            "failed_operations": self._stats.failed_operations,
            "success_rate": self._stats.success_rate,
        }
    
    async def cleanup(self):
        """Cleanup manager resources."""
        await self.stop()




