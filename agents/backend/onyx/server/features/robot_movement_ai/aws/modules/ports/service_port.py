"""
Service Port
============

Port for business logic services.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ServicePort(ABC):
    """Port for business service operations."""
    
    @abstractmethod
    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute business operation."""
        pass
    
    @abstractmethod
    def get_available_operations(self) -> List[str]:
        """Get list of available operations."""
        pass















