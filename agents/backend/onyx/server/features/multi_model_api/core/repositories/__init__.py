"""
Repository pattern for model data access
Abstraction layer for model operations
"""

from .model_repository import ModelRepository
from .registry_repository import RegistryModelRepository

__all__ = [
    "ModelRepository",
    "RegistryModelRepository"
]




