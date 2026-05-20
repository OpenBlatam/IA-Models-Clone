"""
Base Service Handler
===================

Abstract base class for service handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..service_handler import ServiceHandler

from ..service_handler import ServiceType, ServiceConfig, ServiceResult


class BaseServiceHandler(ABC):
    """Abstract base class for service handlers."""
    
    @property
    @abstractmethod
    def service_type(self) -> ServiceType:
        """The service type this handler handles."""
        pass
    
    @property
    @abstractmethod
    def config(self) -> ServiceConfig:
        """Get the service configuration."""
        pass
    
    @abstractmethod
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        """Build the prompt for this service."""
        pass
    
    async def handle(
        self,
        parameters: Dict[str, Any],
        handler: "ServiceHandler"  # Forward reference
    ) -> ServiceResult:
        """
        Handle the service request.
        
        Args:
            parameters: Service parameters
            handler: ServiceHandler instance to execute request
            
        Returns:
            ServiceResult with processing result
        """
        prompt = self.build_prompt(parameters)
        return await handler.execute_request(prompt, self.config)

