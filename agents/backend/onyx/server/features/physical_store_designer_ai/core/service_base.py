"""
Base classes and utilities for services

This module provides base classes for all services in the application,
including common functionality like logging, ID generation, and response formatting.

Example:
    ```python
    from ..core.service_base import BaseService
    
    class MyService(BaseService):
        def __init__(self):
            super().__init__("MyService")
        
        def do_something(self):
            self.log_info("Doing something")
            result = {"status": "success"}
            return self.create_response(result, resource_id="res_123")
    ```
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from .logging_config import get_logger
from .utils import generate_id


class BaseService:
    """Base class for all services with common functionality"""
    
    def __init__(self, service_name: Optional[str] = None):
        self.service_name = service_name or self.__class__.__name__
        self.logger = get_logger(self.service_name)
        self._storage: Dict[str, Any] = {}
    
    def generate_id(self, prefix: str) -> str:
        """Generate a unique ID with prefix"""
        return generate_id(prefix)
    
    def generate_timestamp_id(self, prefix: str) -> str:
        """Generate ID with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{prefix}_{timestamp}"
    
    def create_response(
        self,
        data: Union[Dict[str, Any], Any],
        resource_id: Optional[str] = None,
        note: Optional[str] = None,
        **extra_fields: Any
    ) -> Dict[str, Any]:
        """
        Create a standardized response structure.
        
        Args:
            data: Response data (dict or any object)
            resource_id: Optional resource ID (e.g., "store_123")
            note: Optional note or message
            **extra_fields: Additional fields to include in response
            
        Returns:
            Standardized response dictionary with created_at timestamp
            
        Example:
            ```python
            response = self.create_response(
                {"name": "Store 1"},
                resource_id="store_123",
                note="Store created successfully"
            )
            # Returns: {
            #     "store_id": "store_123",
            #     "name": "Store 1",
            #     "note": "Store created successfully",
            #     "created_at": "2024-01-01T12:00:00"
            # }
            ```
        """
        response: Dict[str, Any] = {
            **extra_fields,
            "created_at": datetime.now().isoformat()
        }
        
        if resource_id:
            # Extract ID type from resource_id prefix
            id_type = resource_id.split("_")[0] if "_" in resource_id else "id"
            response[f"{id_type}_id"] = resource_id
        
        if data:
            if isinstance(data, dict):
                response.update(data)
            else:
                response["data"] = data
        
        if note:
            response["note"] = note
        
        return response
    
    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def log_error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message"""
        self.logger.error(message, extra=kwargs, exc_info=exc_info)
    
    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)


class TimestampedService(BaseService):
    """Service that stores timestamped resources"""
    
    def __init__(self, service_name: Optional[str] = None):
        super().__init__(service_name)
        self._resources: Dict[str, Dict[str, Any]] = {}
    
    def store_resource(self, resource_id: str, resource: Dict[str, Any]) -> None:
        """Store a resource"""
        self._resources[resource_id] = resource
    
    def get_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Get a resource"""
        return self._resources.get(resource_id)
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all resources"""
        return list(self._resources.values())
    
    def delete_resource(self, resource_id: str) -> bool:
        """Delete a resource"""
        if resource_id in self._resources:
            del self._resources[resource_id]
            return True
        return False

