"""
MCP Manifests - Descriptores de recursos
=========================================

Define schemas JSON/YAML para describir recursos MCP:
- Nombre, tipo, permisos
- Endpoints, esquema de datos
- Validación con Pydantic
"""

from .models import ResourceManifest, ResourceType, ResourcePermissions
from .registry import ManifestRegistry
from .loader import ManifestLoader

__all__ = [
    "ResourceManifest",
    "ResourceType",
    "ResourcePermissions",
    "ManifestRegistry",
    "ManifestLoader",
]

