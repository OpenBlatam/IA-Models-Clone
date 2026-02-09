"""
Base Service for Piel Mejorador AI SAM3
=======================================

Base class for services with common functionality.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServiceRequest:
    """Base service request."""
    service_type: str
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ServiceResponse:
    """Base service response."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseService(ABC):
    """
    Base class for services.
    
    Provides common functionality:
    - Request/response handling
    - Error handling
    - Logging
    - Validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base service.
        
        Args:
            config: Optional service configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def execute(self, request: ServiceRequest) -> ServiceResponse:
        """
        Execute service request.
        
        Args:
            request: Service request
            
        Returns:
            Service response
        """
        pass
    
    def validate_request(self, request: ServiceRequest) -> bool:
        """
        Validate service request.
        
        Args:
            request: Service request
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If request is invalid
        """
        if not request.service_type:
            raise ValueError("Service type is required")
        
        if not request.parameters:
            raise ValueError("Parameters are required")
        
        return True
    
    def create_success_response(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ServiceResponse:
        """Create success response."""
        return ServiceResponse(
            success=True,
            data=data,
            metadata=metadata
        )
    
    def create_error_response(
        self,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ServiceResponse:
        """Create error response."""
        return ServiceResponse(
            success=False,
            data={},
            error=error,
            metadata=metadata
        )




