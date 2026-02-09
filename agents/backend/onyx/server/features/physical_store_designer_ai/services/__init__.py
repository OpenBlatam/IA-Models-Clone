"""
Services Module - Modular Organization

Services are organized by category:
- core: Core business services (storage, chat, design)
- ml: Machine learning services
- analysis: Analysis services (competitor, financial, etc.)
- integration: External integrations
- business: Business services (billing, vendors, etc.)

All services are accessible directly from this module for backward compatibility.
"""

# Base classes
from .base import BaseService, TimestampedService, StorageMixin, CacheMixin

# Core services - direct imports for backward compatibility
from .storage_service import StorageService
from .chat_service import ChatService
from .store_designer_service import StoreDesignerService

# Category modules (lazy imports available)
from . import core as core_services
from . import ml as ml_services
from . import analysis as analysis_services
from . import integration as integration_services
from . import business as business_services

# Convenience functions
def get_storage_service():
    """Get storage service instance"""
    return StorageService()

def get_chat_service():
    """Get chat service instance"""
    return ChatService()

def get_store_designer_service():
    """Get store designer service instance"""
    return StoreDesignerService()

__all__ = [
    # Base classes
    "BaseService",
    "TimestampedService",
    "StorageMixin",
    "CacheMixin",
    # Core services
    "StorageService",
    "ChatService",
    "StoreDesignerService",
    # Category modules
    "core_services",
    "ml_services",
    "analysis_services",
    "integration_services",
    "business_services",
    # Convenience functions
    "get_storage_service",
    "get_chat_service",
    "get_store_designer_service",
]
