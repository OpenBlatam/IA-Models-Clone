from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Callable
from ..entities.health import HealthStatus
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Health Service Interface
=======================

Abstract interface for health checking operations.
"""



class IHealthService(ABC):
    """Abstract interface for health check operations."""
    
    @abstractmethod
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        pass
    
    @abstractmethod
    async def run_checks(self) -> HealthStatus:
        """Run all registered health checks."""
        pass
    
    @abstractmethod
    async def check_liveness(self) -> bool:
        """Check if the service is alive."""
        pass
    
    @abstractmethod
    async def check_readiness(self) -> bool:
        """Check if the service is ready to serve requests."""
        pass
    
    @abstractmethod
    async def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        pass 