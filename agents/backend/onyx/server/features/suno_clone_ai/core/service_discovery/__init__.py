"""
Service Discovery Module

Provides:
- Service discovery utilities
- Service registration
- Service lookup
"""

from .discovery import (
    ServiceRegistry,
    register_service,
    discover_service,
    list_services
)

__all__ = [
    "ServiceRegistry",
    "register_service",
    "discover_service",
    "list_services"
]



