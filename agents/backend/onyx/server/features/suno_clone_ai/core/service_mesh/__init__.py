"""
Service Mesh Module

Provides:
- Service mesh utilities
- Inter-service communication
- Mesh networking
"""

from .mesh import (
    ServiceMesh,
    create_mesh,
    register_service,
    call_service
)

__all__ = [
    "ServiceMesh",
    "create_mesh",
    "register_service",
    "call_service"
]



