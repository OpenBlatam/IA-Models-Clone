"""
Service Interface
================

Interface for service implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class IService(ABC):
    """Service interface."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the service."""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the service."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the service."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if service is ready."""
        pass

