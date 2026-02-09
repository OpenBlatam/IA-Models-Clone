"""
Factory classes for creating service instances

This module provides factory classes for creating and managing service instances
with thread-safe singleton patterns for better resource management.
"""

import threading
from typing import Optional, Dict, Any, Type, TypeVar
from .logging_config import get_logger
from ..config.settings import settings

logger = get_logger(__name__)

T = TypeVar('T')


class ServiceFactory:
    """Factory for creating service instances with thread-safe singleton pattern"""
    
    _instances: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_storage_service(cls):
        """Get or create StorageService instance (thread-safe singleton)"""
        if "storage" not in cls._instances:
            with cls._lock:
                if "storage" not in cls._instances:  # Double-check pattern
                    from ..services.storage_service import StorageService
                    cls._instances["storage"] = StorageService()
        return cls._instances["storage"]
    
    @classmethod
    def get_chat_service(cls):
        """Get or create ChatService instance (thread-safe singleton)"""
        if "chat" not in cls._instances:
            with cls._lock:
                if "chat" not in cls._instances:  # Double-check pattern
                    from ..services.chat_service import ChatService
                    cls._instances["chat"] = ChatService()
        return cls._instances["chat"]
    
    @classmethod
    def get_designer_service(cls):
        """Get or create StoreDesignerService instance (thread-safe singleton)"""
        if "designer" not in cls._instances:
            with cls._lock:
                if "designer" not in cls._instances:  # Double-check pattern
                    from ..services.store_designer_service import StoreDesignerService
                    cls._instances["designer"] = StoreDesignerService()
        return cls._instances["designer"]
    
    @classmethod
    def get_service(cls, service_name: str, service_class: Type[T]) -> T:
        """
        Generic method to get or create any service instance (thread-safe singleton)
        
        Args:
            service_name: Unique name for the service
            service_class: Service class to instantiate
            
        Returns:
            Service instance (singleton)
        """
        if service_name not in cls._instances:
            with cls._lock:
                if service_name not in cls._instances:  # Double-check pattern
                    cls._instances[service_name] = service_class()
        return cls._instances[service_name]
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached service instances (thread-safe)"""
        with cls._lock:
            cls._instances.clear()
            logger.debug("Service factory cache cleared")
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """
        Get factory statistics
        
        Returns:
            Dictionary with cached services and instance count
        """
        with cls._lock:
            return {
                "cached_services": list(cls._instances.keys()),
                "total_instances": len(cls._instances)
            }
    
    @classmethod
    def reset_service(cls, service_name: str) -> None:
        """
        Reset a specific service instance (remove from cache)
        
        Args:
            service_name: Name of the service to reset
        """
        with cls._lock:
            if service_name in cls._instances:
                del cls._instances[service_name]
                logger.debug(f"Service '{service_name}' reset from cache")


class ConfigFactory:
    """Factory for creating configuration objects"""
    
    @staticmethod
    def create_app_config() -> Dict[str, Any]:
        """
        Create application configuration dictionary
        
        Returns:
            Dictionary with application settings
        """
        return {
            "host": settings.host,
            "port": settings.port,
            "log_level": settings.log_level,
            "log_format": settings.log_format,
            "environment": settings.environment,
        }
    
    @staticmethod
    def create_cors_config() -> Dict[str, Any]:
        """
        Create CORS configuration dictionary
        
        Returns:
            Dictionary with CORS settings
        """
        return {
            "origins": settings.cors_origins,
            "allow_credentials": settings.cors_allow_credentials,
        }
    
    @staticmethod
    def create_security_config() -> Dict[str, Any]:
        """
        Create security configuration dictionary
        
        Returns:
            Dictionary with security settings
        """
        return {
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "api_key_header": settings.api_key_header,
            "secret_key": "***" if settings.secret_key else None,  # Mask secret
        }

